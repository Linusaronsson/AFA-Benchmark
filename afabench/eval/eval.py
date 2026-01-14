import logging

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

    # Allow classifiers to make predictions using new masked features
    if external_afa_predict_fn is not None:
        external_prediction = external_afa_predict_fn(
            masked_features=new_masked_features,
            feature_mask=new_feature_mask,
            label=label,
            feature_shape=feature_shape,
        )
    else:
        external_prediction = None

    if builtin_afa_predict_fn is not None:
        builtin_prediction = builtin_afa_predict_fn(
            masked_features=new_masked_features,
            feature_mask=new_feature_mask,
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
    selection_budget: int | None = None,
) -> pd.DataFrame:
    """
    Evaluate a single batch.

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
        selection_budget (int|None): How many AFA selections to allow per sample. If None, allow unlimited selections. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - "prev_selections_performed" (list[int]): List of 0-based selection indices performed before this row
            - "selections_performed" (int): Count of how many selections have been performed up to and including this row
            - "builtin_predicted_class" (int|None)
            - "external_predicted_class" (int|None)
            - "true_class" (int)
    """
    # TODO: remove cloning if necessary for speed up
    features = features.clone()
    feature_mask = initial_feature_mask.clone()
    masked_features = initial_masked_features.clone()
    selection_mask = torch.zeros(
        (features.shape[0], n_selection_choices),
        device=features.device,
        dtype=torch.bool,
    )

    # Track which selections have been made per sample (0-based indices)
    selections_performed = [[] for _ in range(features.shape[0])]

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
            label=true_label,
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

        # Append one row per active sample
        for active_idx, true_idx in enumerate(active_indices):
            global_idx = int(true_idx.item())
            action = active_afa_action[active_idx].item()

            # Only selections (action > 0) in the performed list
            if action > 0:
                # Convert action to 0-based selection index
                selection_idx = action - 1
                selections_performed[global_idx].append(selection_idx)

            df_batch_rows.append(
                {
                    "prev_selections_performed": selections_performed[
                        global_idx
                    ][:-1]
                    if action > 0
                    else selections_performed[global_idx],
                    "selections_performed": len(
                        selections_performed[global_idx]
                    ),
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
                }
            )

        # Update feature mask, masked features and selection mask
        masked_features[active_indices] = active_new_masked_features
        feature_mask[active_indices] = active_new_feature_mask
        actions = active_afa_action.squeeze(-1)
        valid_selections = actions > 0
        if valid_selections.any():
            # Update selection mask for valid selections
            selection_mask[
                active_indices[valid_selections], actions[valid_selections] - 1
            ] = True

        # Determine which samples have finished
        # A sample finishes if:
        # - action == 0 (method chose to stop), OR
        # - all features unmasked, OR
        # - selection budget reached
        finished_mask = (actions == 0) | active_new_feature_mask.flatten(
            start_dim=1
        ).all(dim=1)

        # Check if selection budget is reached
        for active_idx, global_idx in enumerate(active_indices):
            global_idx_int = int(global_idx.item())
            if len(selections_performed[global_idx_int]) >= (
                selection_budget or float("inf")
            ):
                finished_mask[active_idx] = True

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

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - "prev_selections_performed" (list[int]): List of 0-based selection indices performed before this row
            - "selections_performed" (int): Count of how many selections have been performed up to and including this row
            - "builtin_predicted_class" (int|None)
            - "external_predicted_class" (int|None)
            - "true_class" (int)
    """
    assert isinstance(dataset, Dataset)
    if device is None:
        device = torch.device("cpu")

    if only_n_samples is not None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(
                (torch.randperm(len(dataset))[:only_n_samples]).tolist()
            ),
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
        )

    batches_df: list[pd.DataFrame] = []
    for _batch_features, _batch_label in tqdm(dataloader):
        batch_features = _batch_features.to(device)
        batch_label = _batch_label.to(device)

        # Initialize masks for the batch
        batch_initial_feature_mask = afa_initialize_fn(
            batch_features,
            batch_label,
            feature_shape=dataset.feature_shape,
        )
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
            )
        )
    # Concatenate all batch DataFrames
    df_batches = pd.concat(batches_df, ignore_index=True)
    # Assert that all the columns described in docstring are present
    expected_columns = {
        "prev_selections_performed",
        "selections_performed",
        "builtin_predicted_class",
        "external_predicted_class",
        "true_class",
    }
    assert expected_columns.issubset(set(df_batches.columns)), (
        f"Expected columns {expected_columns}, but got {set(df_batches.columns)}"
    )
    return df_batches
