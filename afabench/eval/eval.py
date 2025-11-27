import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from afabench.common.afa_initializers.base import AFAInitializer
from afabench.common.custom_types import (
    AFADataset,
    AFAPredictFn,
    AFASelectFn,
    AFASelection,
    AFAUnmaskFn,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)

log = logging.getLogger(__name__)


def single_afa_step(
    features: Features,
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    afa_select_fn: AFASelectFn,
    afa_unmask_fn: AFAUnmaskFn,
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    selection_mask: SelectionMask | None = None,
) -> tuple[
    AFASelection, MaskedFeatures, FeatureMask, Label | None, Label | None
]:
    """
    Perform a single AFA step.

    Args:
        features (Features): True unmasked features, required by unmasker.
        masked_features (MaskedFeatures): Currently masked features.
        feature_mask (FeatureMask): Current feature mask.
        afa_select_fn (AFASelectFn): How to make AFA selections.
        afa_unmask_fn (AFAUnmaskFn): How to select new features from AFA selections.
        external_afa_predict_fn (AFAPredictFn|None): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn|None): A builtin classifier, if such exists.
        selection_mask (SelectionMask|None): Mask indicating which selections have already been performed.

    Returns:
        tuple: (selection, new_masked_features, new_feature_mask, external_pred, builtin_pred)
    """
    # Make AFA selections
    active_afa_selection = afa_select_fn(
        masked_features=masked_features,
        feature_mask=feature_mask,
        # selection_mask=selection_mask,
    )

    # Translate into which unmasked features using the unmasker
    new_masked_features, new_feature_mask = afa_unmask_fn(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features,
        afa_selection=active_afa_selection,
    )

    # Allow classifiers to make predictions using new masked features
    external_prediction = None
    if external_afa_predict_fn is not None:
        external_prediction = external_afa_predict_fn(
            masked_features=new_masked_features,
            feature_mask=new_feature_mask,
        )

    builtin_prediction = None
    if builtin_afa_predict_fn is not None:
        builtin_prediction = builtin_afa_predict_fn(
            masked_features=new_masked_features,
            feature_mask=new_feature_mask,
        )

    return (
        active_afa_selection,
        new_masked_features,
        new_feature_mask,
        external_prediction,
        builtin_prediction,
    )


def process_batch(
    afa_select_fn: AFASelectFn,
    afa_unmask_fn: AFAUnmaskFn,
    n_selection_choices: int,
    features: Features,
    initial_feature_mask: FeatureMask,
    initial_masked_features: MaskedFeatures,
    true_label: Label,
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    selection_budget: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate a single batch.

    Args:
        afa_select_fn: How to make AFA selections. Should return 0 to stop.
        afa_unmask_fn: How to unmask features from AFA selections.
        n_selection_choices: Number of possible AFA selections (excluding 0).
        features: Full features for the batch.
        initial_feature_mask: Initial feature mask for the batch.
        initial_masked_features: Initial masked features for the batch.
        true_label: True labels for the batch.
        external_afa_predict_fn: An external classifier.
        builtin_afa_predict_fn: A builtin classifier, if such exists.
        selection_budget: Max AFA selections per sample. None = unlimited.

    Returns:
        pd.DataFrame with columns for each step taken.
    """
    features = features.clone()
    feature_mask = initial_feature_mask.clone()
    masked_features = initial_masked_features.clone()
    selection_mask = torch.zeros(
        (features.shape[0], n_selection_choices),
        device=features.device,
        dtype=torch.bool,
    )

    selections_performed = [[] for _ in range(features.shape[0])]
    active_indices = torch.arange(features.shape[0], device=features.device)
    df_batch_rows = []

    while len(active_indices) > 0:
        active_features = features[active_indices]
        active_masked_features = masked_features[active_indices]
        active_feature_mask = feature_mask[active_indices]
        active_selection_mask = selection_mask[active_indices]

        (
            active_afa_selection,
            active_new_masked_features,
            active_new_feature_mask,
            active_external_prediction,
            active_builtin_prediction,
        ) = single_afa_step(
            features=active_features,
            masked_features=active_masked_features,
            feature_mask=active_feature_mask,
            afa_select_fn=afa_select_fn,
            afa_unmask_fn=afa_unmask_fn,
            external_afa_predict_fn=external_afa_predict_fn,
            builtin_afa_predict_fn=builtin_afa_predict_fn,
            selection_mask=active_selection_mask,
        )

        # Track selections
        for global_active_idx, afa_selection in zip(
            active_indices, active_afa_selection, strict=True
        ):
            selections_performed[global_active_idx].append(
                afa_selection.item()
            )

        # Record rows
        for active_idx, true_idx in enumerate(active_indices):
            df_batch_rows.append(
                {
                    "feature_indices": active_feature_mask[active_idx]
                    .nonzero(as_tuple=False)
                    .flatten()
                    .cpu()
                    .tolist(),
                    "prev_selections_performed": selections_performed[
                        int(true_idx.item())
                    ][:-1],
                    "selection_performed": active_afa_selection[
                        active_idx
                    ].item(),
                    "next_feature_indices": active_new_feature_mask[active_idx]
                    .nonzero(as_tuple=False)
                    .flatten()
                    .cpu()
                    .tolist(),
                    "builtin_predicted_label": None
                    if active_builtin_prediction is None
                    else active_builtin_prediction[active_idx]
                    .argmax(-1)
                    .item(),
                    "external_predicted_label": None
                    if active_external_prediction is None
                    else active_external_prediction[active_idx]
                    .argmax(-1)
                    .item(),
                    "true_class": true_label[true_idx].argmax(-1).item(),
                }
            )

        # Check finish conditions
        just_finished_mask = (
            active_afa_selection.squeeze(-1) == 0
        ) | active_new_feature_mask.flatten(start_dim=1).all(dim=1)

        # Check budget
        for active_idx, selection_list in enumerate(selections_performed):
            if len(selection_list) >= (selection_budget or float("inf")):
                just_finished_mask[active_idx] = True

        # Update state
        masked_features[active_indices] = active_new_masked_features
        feature_mask[active_indices] = active_new_feature_mask
        sel = active_afa_selection.squeeze(-1)
        valid = sel > 0
        if valid.any():
            selection_mask[active_indices[valid], sel[valid] - 1] = True

        # Filter finished
        active_indices = active_indices[~just_finished_mask]

    return pd.DataFrame(df_batch_rows)


def eval_afa_method(
    afa_select_fn: AFASelectFn,
    afa_unmask_fn: AFAUnmaskFn,
    n_selection_choices: int,
    afa_initializer: AFAInitializer,
    dataset: AFADataset,
    external_afa_predict_fn: AFAPredictFn | None = None,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    only_n_samples: int | None = None,
    device: torch.device | None = None,
    selection_budget: int | None = None,
    batch_size: int = 1,
    n_initial_features: int = 0,
) -> pd.DataFrame:
    """
    Evaluate an AFA method with support for early stopping and batched processing.

    Args:
        afa_select_fn: How to choose AFA actions. Should return 0 to stop.
        afa_unmask_fn: How to unmask features from AFA actions.
        n_selection_choices: Number of possible AFA selections (excluding 0).
        afa_initializer: How to create initial feature mask.
        dataset: The dataset to evaluate on.
        external_afa_predict_fn: An external classifier.
        builtin_afa_predict_fn: A builtin classifier, if such exists.
        only_n_samples: If specified, only evaluate on this many samples.
        device: Device to place data on. Defaults to "cpu".
        selection_budget: Max selections per sample. None = unlimited.
        batch_size: Batch size for processing.
        n_initial_features: How many features to reveal at initialization.

    Returns:
        pd.DataFrame with evaluation results.
    """
    assert isinstance(dataset, Dataset)
    device = device or torch.device("cpu")

    if only_n_samples is not None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(
                (torch.randperm(len(dataset))[:only_n_samples]).tolist()
            ),
        )
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size)

    df_batches: list[pd.DataFrame] = []

    for batch_features, batch_label in tqdm(dataloader):
        batch_features = batch_features.to(device)
        batch_label = batch_label.to(device)

        # Initialize using the initializer
        batch_masked_features, batch_feature_mask = afa_initializer.initialize(
            features=batch_features,
            n_features_select=n_initial_features,
        )

        df_batches.append(
            process_batch(
                afa_select_fn=afa_select_fn,
                afa_unmask_fn=afa_unmask_fn,
                n_selection_choices=n_selection_choices,
                features=batch_features,
                initial_feature_mask=batch_feature_mask,
                initial_masked_features=batch_masked_features,
                true_label=batch_label,
                external_afa_predict_fn=external_afa_predict_fn,
                builtin_afa_predict_fn=builtin_afa_predict_fn,
                selection_budget=selection_budget,
            )
        )

    return pd.concat(df_batches, ignore_index=True)
