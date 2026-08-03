import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast, final, override

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from afabench.core.types import (
    AFAAction,
    AFAActionFn,
    AFADataset,
    AFAInitializeFn,
    AFAPredictFn,
    AFASelection,
    AFAUnmaskFn,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)

log = logging.getLogger(__name__)


class _TensorDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...


@final
class _DatasetWithAvailability(
    Dataset[tuple[torch.Tensor, torch.Tensor, FeatureMask, SelectionMask]]
):
    def __init__(
        self,
        dataset: _TensorDataset,
        feature_availability: FeatureMask,
        selection_availability: SelectionMask,
    ) -> None:
        self._dataset = dataset
        self._feature_availability = feature_availability
        self._selection_availability = selection_availability

    def __len__(self) -> int:
        return len(self._dataset)

    @override
    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, FeatureMask, SelectionMask]:
        features, label = self._dataset[idx]
        return (
            features,
            label,
            self._feature_availability[idx],
            self._selection_availability[idx],
        )


def _initial_eval_state(
    features: Features,
    initial_feature_mask: FeatureMask,
    initial_masked_features: MaskedFeatures,
    n_selection_choices: int,
    feature_availability: FeatureMask | None,
    selection_availability: SelectionMask | None,
) -> tuple[FeatureMask, MaskedFeatures, SelectionMask]:
    feature_mask = initial_feature_mask.clone()
    if feature_availability is not None:
        if feature_availability.shape != features.shape:
            msg = "feature_availability must have the same shape as features."
            raise ValueError(msg)
        feature_mask &= feature_availability
    masked_features = initial_masked_features.clone()
    masked_features[~feature_mask] = 0.0

    expected_selection_shape = (features.shape[0], n_selection_choices)
    if selection_availability is None:
        selection_mask = torch.zeros(
            expected_selection_shape,
            device=features.device,
            dtype=torch.bool,
        )
    else:
        if selection_availability.shape != expected_selection_shape:
            msg = (
                "selection_availability must have shape "
                f"{expected_selection_shape}."
            )
            raise ValueError(msg)
        selection_mask = ~selection_availability.clone()
    return feature_mask, masked_features, selection_mask


def _dataset_with_availability(
    dataset: AFADataset,
    feature_availability: FeatureMask | None,
    selection_availability: SelectionMask | None,
) -> Dataset[tuple[torch.Tensor, ...]]:
    if (feature_availability is None) != (selection_availability is None):
        msg = (
            "feature_availability and selection_availability must be supplied "
            "together."
        )
        raise ValueError(msg)
    if feature_availability is None:
        return cast(
            "Dataset[tuple[torch.Tensor, ...]]", cast("object", dataset)
        )
    assert selection_availability is not None
    if len(feature_availability) != len(dataset):
        msg = "feature_availability must contain one row per dataset item."
        raise ValueError(msg)
    if len(selection_availability) != len(dataset):
        msg = "selection_availability must contain one row per dataset item."
        raise ValueError(msg)
    tensor_dataset = cast("_TensorDataset", cast("object", dataset))
    return _DatasetWithAvailability(
        tensor_dataset,
        feature_availability,
        selection_availability,
    )


def override_stop_with_first_selection(
    afa_action: AFAAction, selection_mask: SelectionMask
) -> AFAAction:
    """Override each stop action in `afa_action` with the first available selection in `selection_mask`."""
    new_afa_action = afa_action.clone()

    flat_selection_mask = selection_mask.bool().flatten(start_dim=1)

    available = ~flat_selection_mask
    first_available_selection = available.int().argmax(dim=1)
    mask = (afa_action == 0).squeeze(-1) & available.any(dim=1)
    new_afa_action[mask] = (
        (first_available_selection[mask] + 1)
        .to(new_afa_action.dtype)
        .unsqueeze(-1)
    )

    return new_afa_action


@dataclass
class AFAStepResult:
    action: AFAAction  # which action was chosen
    masked_features: MaskedFeatures  # new masked features
    feature_mask: FeatureMask  # new feature mask
    external_prediction: Label | None
    builtin_prediction: Label | None
    acquisition_forced: (
        torch.Tensor
    )  # bool tensor, true if acquisition was forced


def single_afa_step(
    features: Features,
    label: Label,
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    selection_mask: SelectionMask,
    afa_action_fn: AFAActionFn,
    afa_unmask_fn: AFAUnmaskFn,
    feature_shape: torch.Size | None = None,
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    *,
    force_acquisition: bool = False,
    enforce_selection_mask: bool = False,
) -> AFAStepResult:
    """
    Perform a single AFA step.

    Args:
        label: Passed to all functions that may need it. Normally unused
            (accessing it would be cheating), but available for benchmarking.
        feature_shape: Required by some action/unmask/predict implementations.
        force_acquisition: if true, override stop actions with the next available acquisition action instead. Useful in the hard budget setting.
        enforce_selection_mask: reject acquisitions that are unavailable or
            already masked. Native-missingness evaluation enables this so an
            imputed source value can never be revealed as a measurement.
    """
    # Get the action from the AFA method (0 = stop, 1-n = valid selections)
    afa_action = afa_action_fn(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        label=label,
        feature_shape=feature_shape,
    )
    batch_size = afa_action.shape[:-1]

    acquisition_forced = torch.zeros(batch_size, dtype=torch.bool)
    if force_acquisition:
        overriden_afa_action = override_stop_with_first_selection(
            afa_action, selection_mask
        )
        acquisition_forced[
            (overriden_afa_action != afa_action).squeeze(-1)
        ] = True
        afa_action = overriden_afa_action

    if enforce_selection_mask:
        actions = afa_action.squeeze(-1)
        is_selection = actions > 0
        if is_selection.any():
            selected = actions[is_selection] - 1
            if (selected >= selection_mask.shape[-1]).any() or selection_mask[
                is_selection, selected
            ].any():
                msg = "AFA method selected an unavailable acquisition."
                raise ValueError(msg)

    # Convert action to selection for the unmasker
    afa_selection: AFASelection = afa_action - 1
    no_stop_mask = afa_action.squeeze(-1) != 0
    # Only give non-stopping samples to the unmasker
    new_feature_mask_no_stop = afa_unmask_fn(
        masked_features=masked_features[no_stop_mask],
        feature_mask=feature_mask[no_stop_mask],
        features=features[no_stop_mask],
        afa_selection=afa_selection[no_stop_mask],
        selection_mask=selection_mask[no_stop_mask],
        label=label[no_stop_mask],
        feature_shape=feature_shape,
    )
    new_feature_mask = feature_mask.clone()
    new_feature_mask[no_stop_mask] = new_feature_mask_no_stop
    new_masked_features = features.clone()
    new_masked_features[~new_feature_mask] = 0.0

    # Allow classifiers to make predictions using **current** features
    if external_afa_predict_fn is not None:
        external_prediction = external_afa_predict_fn(
            masked_features=masked_features,
            feature_mask=feature_mask,
            label=label,
            feature_shape=feature_shape,
        )
    else:
        external_prediction = None

    if builtin_afa_predict_fn is not None:
        builtin_prediction = builtin_afa_predict_fn(
            masked_features=masked_features,
            feature_mask=feature_mask,
            label=label,
            feature_shape=feature_shape,
        )
    else:
        builtin_prediction = None

    return AFAStepResult(
        action=afa_action,
        masked_features=new_masked_features,
        feature_mask=new_feature_mask,
        external_prediction=external_prediction,
        builtin_prediction=builtin_prediction,
        acquisition_forced=acquisition_forced,
    )


@dataclass
class _RowBuffers:
    """
    One entry per timestep, concatenated once when the batch finishes.

    Everything stays on device until then, so assembling the result frame costs
    a single device-to-host transfer for the whole batch rather than a handful
    per active sample per timestep.
    """

    idx: list[torch.Tensor] = field(default_factory=list)
    action: list[torch.Tensor] = field(default_factory=list)
    cost: list[torch.Tensor] = field(default_factory=list)
    forced: list[torch.Tensor] = field(default_factory=list)
    builtin: list[torch.Tensor] = field(default_factory=list)
    external: list[torch.Tensor] = field(default_factory=list)

    def to_frame(self, true_label: Label, n_samples: int) -> pd.DataFrame:
        idx = torch.cat(self.idx).cpu().tolist()
        action = torch.cat(self.action).cpu().tolist()
        n_rows = len(idx)

        def classes(buffer: list[torch.Tensor]) -> list[int | None]:
            # A predict_fn that was never supplied yields a column of None,
            # which is the object dtype callers already expect.
            if not buffer:
                return [None] * n_rows
            return torch.cat(buffer).cpu().tolist()

        return pd.DataFrame(
            {
                "prev_selections_performed": _prev_selections(
                    idx, action, n_samples
                ),
                "action_performed": action,
                "builtin_predicted_class": classes(self.builtin),
                "external_predicted_class": classes(self.external),
                "true_class": true_label.argmax(-1).cpu()[idx].tolist(),
                "accumulated_cost": torch.cat(self.cost).cpu().tolist(),
                "idx": idx,
                "forced_stop": torch.cat(self.forced).cpu().tolist(),
            }
        )


def _prev_selections(
    row_idx: list[int], row_action: list[int], n_samples: int
) -> list[list[int]]:
    """
    Rebuild, per row, the selections that sample had made before this timestep.

    Rows arrive timestep-major and each sample appears at most once per
    timestep, so one forward walk suffices: what a sample has accumulated when
    its row is reached is exactly that row's history. Reconstructing this at the
    end keeps the acquisition loop free of per-sample Python bookkeeping.
    """
    running: list[list[int]] = [[] for _ in range(n_samples)]
    out: list[list[int]] = []
    for global_idx, action in zip(row_idx, row_action, strict=True):
        out.append(running[global_idx].copy())
        # Stop actions append -1, which is never read back: a row's history
        # covers strictly earlier timesteps, and a sample stops only once.
        running[global_idx].append(action - 1)
    return out


def process_batch(
    afa_action_fn: AFAActionFn,
    afa_unmask_fn: AFAUnmaskFn,
    n_selection_choices: int,
    features: Features,
    initial_feature_mask: FeatureMask,
    initial_masked_features: MaskedFeatures,
    true_label: Label,
    feature_shape: torch.Size | None = None,
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    selection_budget: float | None = None,
    selection_costs: Sequence[float] | None = None,
    *,
    force_acquisition: bool = False,
    feature_availability: FeatureMask | None = None,
    selection_availability: SelectionMask | None = None,
) -> pd.DataFrame:
    """
    Evaluate a single batch.

    Until every sample either
    1. Stops feature acquisition voluntarily (action=0).
    2. Runs out of selection budget and is forced to stop (action=0). More precisely, this means that if the next action was selected such that the budget would have been **exceeded**, we instead force the stop action.

    Assumes that predictions are for classes, and only stores the most likely class prediction.

    Args:
        afa_action_fn (AFAActionFn): How to make AFA actions. Should return 0 to stop or 1-n to select.
        afa_unmask_fn (AFAUnmaskFn): How to select new features from AFA selections.
        n_selection_choices (int): Number of possible AFA selections (excluding stop action). Should reflect the AFAUnmaskFn behavior.
        features (Features): Features for the batch.
        initial_feature_mask (FeatureMask): Initial feature mask for the batch.
        initial_masked_features (MaskedFeatures): Initial masked features for the batch.
        true_label (torch.Tensor): True labels for the batch.
        feature_shape (torch.Size|None): Shape of the features, required by some objects.
        external_afa_predict_fn (AFAPredictFn): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn): A builtin classifier, if such exists.
        selection_budget (float|None): Total accumulated selection cost to allow per sample. If None, allow unlimited selections. Defaults to None.
        selection_costs (Sequence[float]|None): How much each selection costs. If not provided, assume unit cost (1) for each selection.
        force_acquisition (bool): Whether to force feature acquisition, ignoring stop actions.
        feature_availability: Per-sample features that exist in the source
            data. Initial features are intersected with this mask.
        selection_availability: Per-sample acquisitions that are legal. False
            selections are presented to the policy as permanently masked.

    Returns:
        pd.DataFrame: DataFrame with one row per sample and timestep, containing columns:
            - "prev_selections_performed" (list[int]): List of 0-based selection indices that the method had previously performed up to the given timestep. The length of this list gives the timestep of the current episode.
            - "action_performed" (int): Which action the method chose.
            - "builtin_predicted_class" (int|None)
            - "external_predicted_class" (int|None)
            - "true_class" (int)
            - "accumulated_cost" (float): Acculumulated cost from `prev_selections_performed` **and** the current action.
            - "idx" (int): Which sample the row corresponds to.
            - "forced_stop" (bool): Whether the episode terminated due to budget being exceeded.
    """
    # TODO: remove cloning if necessary for speed up
    features = features.clone()
    feature_mask, masked_features, selection_mask = _initial_eval_state(
        features,
        initial_feature_mask,
        initial_masked_features,
        n_selection_choices,
        feature_availability,
        selection_availability,
    )

    n_samples = features.shape[0]
    device = features.device

    # Per-sample state, kept on device so the acquisition loop never syncs to
    # read it back. float64 because these accumulate one addend per timestep and
    # the budget comparison below has to be exact.
    accumulated_costs = torch.zeros(
        n_samples, dtype=torch.float64, device=device
    )
    forced_stops = torch.zeros(n_samples, dtype=torch.bool, device=device)
    selection_costs_t = (
        torch.ones(n_selection_choices, dtype=torch.float64, device=device)
        if selection_costs is None
        else torch.tensor(
            list(selection_costs), dtype=torch.float64, device=device
        )
    )

    # Process a subset of the batch, which gets smaller and smaller until it's empty
    active_indices = torch.arange(n_samples, device=device)

    rows = _RowBuffers()

    while len(active_indices) > 0:
        active_features = features[active_indices]
        active_masked_features = masked_features[active_indices]
        active_feature_mask = feature_mask[active_indices]
        active_selection_mask = selection_mask[active_indices]

        step = single_afa_step(
            features=active_features,
            label=true_label[active_indices],
            masked_features=active_masked_features,
            feature_mask=active_feature_mask,
            afa_action_fn=afa_action_fn,
            afa_unmask_fn=afa_unmask_fn,
            feature_shape=feature_shape,
            external_afa_predict_fn=external_afa_predict_fn,
            builtin_afa_predict_fn=builtin_afa_predict_fn,
            selection_mask=active_selection_mask,
            force_acquisition=force_acquisition,
            enforce_selection_mask=selection_availability is not None,
        )
        # Key assumption: predictions are logits/probabilities for classes
        if step.builtin_prediction is not None:
            assert step.builtin_prediction.shape[-1] > 1, (
                "Expected builtin prediction to have class dimension"
            )
        if step.external_prediction is not None:
            assert step.external_prediction.shape[-1] > 1, (
                "Expected external prediction to have class dimension"
            )

        # `actions` is a view, so overriding it below also overrides
        # `step.action`, matching the order the unbatched version relied on.
        actions = step.action.squeeze(-1)

        # Check budget and override actions if they would exceed it
        if selection_budget is not None:
            is_selection = actions > 0
            # clamp_min keeps the gather in range for stop actions, whose cost
            # is discarded by `is_selection` anyway.
            action_costs = selection_costs_t[(actions - 1).clamp_min(0)]
            exceeds = is_selection & (
                accumulated_costs[active_indices] + action_costs
                > selection_budget
            )
            actions[exceeds] = 0
            forced_stops[active_indices[exceeds]] = True

        # Update accumulated costs for valid selections (BEFORE appending rows)
        valid_selections = actions > 0
        valid_indices = active_indices[valid_selections]
        accumulated_costs.index_add_(
            0, valid_indices, selection_costs_t[actions[valid_selections] - 1]
        )

        # One row per active sample, kept on device and assembled at the end.
        rows.idx.append(active_indices.clone())
        rows.action.append(actions.clone())
        rows.cost.append(accumulated_costs[active_indices])
        rows.forced.append(forced_stops[active_indices])
        if step.builtin_prediction is not None:
            rows.builtin.append(step.builtin_prediction.argmax(-1))
        if step.external_prediction is not None:
            rows.external.append(step.external_prediction.argmax(-1))

        # Update feature mask, masked features and selection mask
        masked_features[active_indices] = step.masked_features
        feature_mask[active_indices] = step.feature_mask
        # Empty index tensors make this a no-op, so it needs no guard -- and a
        # guard would cost a device sync per timestep to evaluate.
        selection_mask[valid_indices, actions[valid_selections] - 1] = True

        # A sample finishes if action == 0, either by manually choosing it or being forced to choose it due to exceeding the budget. Notably, we don't care whether you've already choosen all features and choose to do more selections. This scenario could be interesting with stochastic unmaskers.
        finished_mask = actions == 0

        # Filter out finished samples
        active_indices = active_indices[~finished_mask]

    return rows.to_frame(true_label, n_samples)


def eval_afa_method(
    afa_action_fn: AFAActionFn,
    afa_unmask_fn: AFAUnmaskFn,
    n_selection_choices: int,
    afa_initialize_fn: AFAInitializeFn,
    dataset: AFADataset,  # we also check at runtime that this is a pytorch Dataset
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    only_n_samples: int | None = None,
    device: torch.device | None = None,
    selection_budget: int | None = None,
    batch_size: int = 1,
    selection_costs: Sequence[float] | None = None,
    seed: int | None = None,
    *,
    force_acquisition: bool = False,
    feature_availability: FeatureMask | None = None,
    selection_availability: SelectionMask | None = None,
) -> pd.DataFrame:
    """
    Evaluate an AFA method with support for early stopping and batched processing.

    Args:
        afa_action_fn (AFAActionFn): How to choose AFA actions. Should return 0 to stop or 1-n to select.
        afa_unmask_fn (AFAUnmaskFn): How to select new features from AFA actions.
        n_selection_choices (int): Number of possible AFA selections (excluding 0).
        afa_initialize_fn (AFAInitializeFn): How to create the initial feature mask.
        dataset (AFADataset & Dataset): The dataset to evaluate on. Additionally assumed to be a torch dataset.
        external_afa_predict_fn (AFAPredictFn): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn): A builtin classifier, if such exists.
        only_n_samples (int|None, optional): If specified, only evaluate on this many samples from the dataset. Defaults to None.
        device (torch.device|None): Device to place data on. Defaults to "cpu".
        selection_budget (int|None): How many AFA selections to allow per sample. If None, allow unlimited selections. Defaults to None.
        batch_size (int): Batch size for processing samples. Defaults to 1.
        selection_costs (Sequence[float]|None): How much each selection costs. If not provided, assume unit cost (1) for each selection.
        seed (int|None): Seed for evaluation-time sampling, such as `only_n_samples`.
        force_acquisition (bool): Whether to force feature acquisition, ignoring stop actions.
        feature_availability: Per-sample feature support in the source data.
        selection_availability: Legal selections derived from feature support.

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - "prev_selections_performed" (list[int]): List of 0-based selection indices performed before this row
            - "action_performed" (int): Which action the method chose.
            - "builtin_predicted_class" (int|None)
            - "external_predicted_class" (int|None)
            - "true_class" (int)
            - "forced_stop" (bool): Whether stopping happened due to exceeding the budget.
    """
    assert isinstance(dataset, Dataset)
    if device is None:
        device = torch.device("cpu")

    loader_dataset = _dataset_with_availability(
        dataset,
        feature_availability,
        selection_availability,
    )

    sampler_generator = None
    if seed is not None:
        sampler_generator = torch.Generator().manual_seed(seed)

    if only_n_samples is not None:
        sample_indices = torch.randperm(
            len(dataset), generator=sampler_generator
        )[:only_n_samples].tolist()
        dataloader = DataLoader(
            loader_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(
                sample_indices, generator=sampler_generator
            ),
        )
    else:
        dataloader = DataLoader(
            loader_dataset,
            batch_size=batch_size,
        )

    batches_df: list[pd.DataFrame] = []
    # Nothing here is differentiated, but several methods build a graph anyway.
    # DIME's `act` and `predict` in particular carry no internal guard, so
    # without this every acquisition step allocates and discards one.
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            _batch_features, _batch_label = batch[:2]
            batch_features = _batch_features.to(device)
            batch_label = _batch_label.to(device)
            batch_feature_availability = (
                batch[2].to(device)
                if feature_availability is not None
                else None
            )
            batch_selection_availability = (
                batch[3].to(device)
                if selection_availability is not None
                else None
            )

            # Initialize masks for the batch
            batch_initial_feature_mask = afa_initialize_fn(
                batch_features,
                batch_label,
                feature_shape=dataset.feature_shape,
            ).to(device)
            batch_initial_masked_features = batch_features.clone()
            batch_initial_masked_features[~batch_initial_feature_mask] = (
                0.0  # Assuming zero masking
            )

            batches_df.append(
                process_batch(
                    afa_action_fn=afa_action_fn,
                    afa_unmask_fn=afa_unmask_fn,
                    n_selection_choices=n_selection_choices,
                    features=batch_features,
                    initial_feature_mask=batch_initial_feature_mask,
                    initial_masked_features=batch_initial_masked_features,
                    true_label=batch_label,
                    feature_shape=dataset.feature_shape,
                    external_afa_predict_fn=external_afa_predict_fn,
                    builtin_afa_predict_fn=builtin_afa_predict_fn,
                    selection_budget=selection_budget,
                    selection_costs=selection_costs,
                    force_acquisition=force_acquisition,
                    feature_availability=batch_feature_availability,
                    selection_availability=batch_selection_availability,
                )
            )
    # Concatenate all batch DataFrames
    df_batches = pd.concat(batches_df, ignore_index=True)
    # Assert that all the columns described in docstring are present
    expected_columns = {
        "prev_selections_performed",
        "action_performed",
        "builtin_predicted_class",
        "external_predicted_class",
        "true_class",
        "forced_stop",
    }
    assert expected_columns.issubset(set(df_batches.columns)), (
        f"Expected columns {expected_columns}, but got {set(df_batches.columns)}"
    )
    return df_batches
