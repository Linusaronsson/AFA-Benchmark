from __future__ import annotations

import logging
from collections.abc import Callable, Sequence  # noqa: TC003
from typing import override

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from afabench.common.custom_types import (
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


class _DatasetWithIndex(Dataset[tuple[int, Features, Label]]):
    """Wrap a dataset so evaluation can retain dataset-level sample ids."""

    def __init__(self, dataset: AFADataset) -> None:
        self.dataset: AFADataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    @override
    def __getitem__(self, idx: int) -> tuple[int, Features, Label]:
        features, label = self.dataset[idx]
        return idx, features, label


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
) -> tuple[AFAAction, MaskedFeatures, FeatureMask, Label | None, Label | None]:
    """
    Perform a single AFA step.

    Args:
        features (Features): True unmasked features, required by unmasker.
        label (Label): The true label, passed to all functions that may need it. Usually not used, since that would be a form of cheating, but we might want some objects to have access to it for benchmarking.
        masked_features (MaskedFeatures): Currently masked features.
        feature_mask (FeatureMask): Current feature mask.
        selection_mask (SelectionMask): Mask indicating which selections have already been performed.
        afa_action_fn (AFAActionFn): How to make AFA actions. Returns 0 to stop or 1-n to select.
        afa_unmask_fn (AFAUnmaskFn): How to select new features from AFA selections.
        feature_shape (torch.Size|None): Shape of the features, required by some objects.
        external_afa_predict_fn (AFAPredictFn|None): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn|None): A builtin classifier, if such exists.

    Returns:
        tuple[AFAAction, MaskedFeatures, FeatureMask, Label|None, Label|None]: Action made (0=stop, 1-n=selection), updated masked features, feature mask and predicted labels after the AFA step.
    """
    # Get the action from the AFA method (0 = stop, 1-n = valid selections)
    afa_action = afa_action_fn(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        label=label,
        feature_shape=feature_shape,
    )

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

    return (
        afa_action,
        new_masked_features,
        new_feature_mask,
        external_prediction,
        builtin_prediction,
    )


def process_batch(  # noqa: C901, PLR0912, PLR0915
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
    initial_selection_mask: SelectionMask | None = None,
    sample_ids: Sequence[int] | None = None,
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
        initial_selection_mask (SelectionMask|None): Pre-initialized selection mask marking features as already forbidden (True=forbidden). If None, starts with all features acquirable.
        sample_ids (Sequence[int]|None): Dataset-level sample ids to attach to rows.
            If None, uses 0..batch_size-1.

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
    feature_mask = initial_feature_mask.clone()
    masked_features = initial_masked_features.clone()
    if initial_selection_mask is not None:
        assert initial_selection_mask.shape[-1] == n_selection_choices, (
            "initial_selection_mask must be in selection space. "
            f"Expected trailing dim {n_selection_choices}, got "
            f"{initial_selection_mask.shape[-1]}."
        )
        selection_mask = initial_selection_mask.clone()
    else:
        selection_mask = torch.zeros(
            (features.shape[0], n_selection_choices),
            device=features.device,
            dtype=torch.bool,
        )
    resolved_sample_ids = (
        list(range(features.shape[0]))
        if sample_ids is None
        else [int(sample_id) for sample_id in sample_ids]
    )
    assert len(resolved_sample_ids) == features.shape[0], (
        "sample_ids must have one entry per sample in the batch."
    )

    # Track which selections have been made per sample (0-based indices)
    selections_performed = [[] for _ in range(features.shape[0])]

    # Track accumulated costs per sample
    accumulated_costs = [0.0 for _ in range(features.shape[0])]

    # Track whether each sample was forced to stop due to budget
    forced_stops = [False for _ in range(features.shape[0])]

    # Convert selection_costs to a list if provided, otherwise use unit costs
    if selection_costs is None:
        selection_costs_list = [1.0] * n_selection_choices
    else:
        selection_costs_list = list(selection_costs)

    # Process a subset of the batch, which gets smaller and smaller until it's empty
    active_indices = torch.arange(features.shape[0], device=features.device)

    df_batch_rows = []

    while len(active_indices) > 0:
        active_features = features[active_indices]
        active_masked_features = masked_features[active_indices]
        active_feature_mask = feature_mask[active_indices]
        active_selection_mask = selection_mask[active_indices]

        (
            active_afa_action,
            active_new_masked_features,
            active_new_feature_mask,
            active_external_prediction,
            active_builtin_prediction,
        ) = single_afa_step(
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
        )
        # Key assumption: predictions are logits/probabilities for classes
        if active_builtin_prediction is not None:
            assert active_builtin_prediction.shape[-1] > 1, (
                "Expected builtin prediction to have class dimension"
            )
        if active_external_prediction is not None:
            assert active_external_prediction.shape[-1] > 1, (
                "Expected external prediction to have class dimension"
            )

        # Check budget and override actions if they would exceed it
        if selection_budget is not None:
            for active_idx, true_idx in enumerate(active_indices):
                global_idx = int(true_idx.item())
                action = int(active_afa_action[active_idx].item())
                if action > 0:  # Valid selection (not stop action)
                    selection_idx = action - 1
                    action_cost = selection_costs_list[selection_idx]
                    if (
                        accumulated_costs[global_idx] + action_cost
                        > selection_budget
                    ):
                        # Override action to stop (0) if it would exceed budget
                        active_afa_action[active_idx, 0] = 0
                        forced_stops[global_idx] = True

        # Update accumulated costs for valid selections (BEFORE appending rows)
        actions = active_afa_action.squeeze(-1)
        valid_selections = actions > 0
        if valid_selections.any():
            for active_idx, global_idx in enumerate(active_indices):
                global_idx_int = int(global_idx.item())
                action = int(actions[active_idx].item())
                if action > 0:  # Valid selection (not stop action)
                    selection_idx = action - 1
                    accumulated_costs[global_idx_int] += selection_costs_list[
                        selection_idx
                    ]

        # Append one row per active sample
        for active_idx, true_idx in enumerate(active_indices):
            global_idx = int(true_idx.item())
            action = active_afa_action[active_idx].item()
            # It does not matter if we append -1 here, since we will never access the last value
            selections_performed[global_idx].append(action - 1)

            df_batch_rows.append(
                {
                    "prev_selections_performed": (
                        selections_performed[global_idx][:-1]
                    ).copy(),
                    "action_performed": action,
                    "builtin_predicted_class": None
                    if active_builtin_prediction is None
                    else active_builtin_prediction[active_idx]
                    .argmax(-1)
                    .item(),
                    "external_predicted_class": None
                    if active_external_prediction is None
                    else active_external_prediction[active_idx]
                    .argmax(-1)
                    .item(),
                    "true_class": true_label[true_idx].argmax(-1).item(),
                    "accumulated_cost": accumulated_costs[global_idx],
                    "idx": resolved_sample_ids[global_idx],
                    "forced_stop": forced_stops[global_idx],
                }
            )

        # Update feature mask, masked features and selection mask
        masked_features[active_indices] = active_new_masked_features
        feature_mask[active_indices] = active_new_feature_mask
        if valid_selections.any():
            # Update selection mask for valid selections
            selection_mask[
                active_indices[valid_selections], actions[valid_selections] - 1
            ] = True

        # A sample finishes if action == 0, either by manually choosing it or being forced to choose it due to exceeding the budget. Notably, we don't care whether you've already choosen all features and choose to do more selections. This scenario could be interesting with stochastic unmaskers.
        finished_mask = actions == 0

        # Filter out finished samples
        active_indices = active_indices[~finished_mask]

    return pd.DataFrame(df_batch_rows)


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
    forbidden_mask_fn: Callable[[FeatureMask, torch.Size], SelectionMask]
    | None = None,
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
        forbidden_mask_fn (Callable|None): If provided, called with (observed_mask, feature_shape) to produce an initial SelectionMask marking never-acquirable features as forbidden.

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

    if only_n_samples is not None:
        indexed_dataset = _DatasetWithIndex(dataset)
        dataloader = DataLoader(
            indexed_dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(
                (torch.randperm(len(dataset))[:only_n_samples]).tolist()
            ),
        )
    else:
        indexed_dataset = _DatasetWithIndex(dataset)
        dataloader = DataLoader(
            indexed_dataset,
            batch_size=batch_size,
        )

    batches_df: list[pd.DataFrame] = []
    for batch_idx, _batch_features, _batch_label in tqdm(
        dataloader, desc="Evaluating AFA"
    ):
        batch_features = _batch_features.to(device)
        batch_label = _batch_label.to(device)

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

        # Compute forbidden selection mask if a function is provided
        batch_initial_selection_mask = None
        if forbidden_mask_fn is not None:
            batch_initial_selection_mask = forbidden_mask_fn(
                batch_initial_feature_mask, dataset.feature_shape
            ).to(device)

        df_batch = process_batch(
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
            initial_selection_mask=batch_initial_selection_mask,
            sample_ids=batch_idx.tolist(),
        )
        batches_df.append(df_batch)
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
