import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.custom_types import (
    AFADataset,
    AFAPredictFn,
    AFASelectFn,
)

log = logging.getLogger(__name__)


class _MaskLayer2d(torch.nn.Module):
    def __init__(self, mask_width: int, patch_size: int):
        super().__init__()
        self.mask_width = mask_width
        self.patch_size = patch_size
        self.upsample = (
            torch.nn.Identity()
            if patch_size == 1
            else torch.nn.Upsample(scale_factor=patch_size)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if len(mask.shape) == 2:
            B, N = mask.shape
            mask = mask.view(B, 1, self.mask_width, self.mask_width)
        elif mask.dim() != 4:
            raise ValueError(f"Unexpected mask shape {tuple(mask.shape)}")
        m = self.upsample(mask)
        return x * m


def eval_soft_budget_afa_method(
    afa_select_fn: AFASelectFn,
    dataset: AFADataset,  # also assumed to subclass torch Dataset
    external_afa_predict_fn: AFAPredictFn,
    builtin_afa_predict_fn: AFAPredictFn | None = None,
    only_n_samples: int | None = None,
    device: torch.device | None = None,
    batch_size: int = 1,
) -> pd.DataFrame:
    """
    Evaluate an AFA method with support for early stopping and batched processing.

    Args:
        afa_select_fn (AFASelectFn): How to select new features. Should return 0 to stop.
        method_name (str): Name of the method, included in results DataFrame.
        dataset (AFADataset): The dataset to evaluate on. Additionally assumed to be a torch dataset.
        external_afa_predict_fn (AFAPredictFn): An external classifier.
        builtin_afa_predict_fn (AFAPredictFn): A builtin classifier, if such exists.
        only_n_samples (int|None, optional): If specified, only evaluate on this many samples from the dataset. Defaults to None.
        device (torch.device|None): Device to place data on. Defaults to "cpu".
        batch_size (int): Batch size for processing samples. Defaults to 1.

    Returns:
        pd.DataFrame: DataFrame containing columns:
            - "features_chosen"
            - "predicted_label_builtin"
            - "predicted_label_external"
    """
    if device is None:
        device = torch.device("cpu")

    data_rows = []

    acquisition_costs = dataset.get_feature_acquisition_costs().to(device)
    log.debug(f"Acquisition costs: {acquisition_costs}")

    # Optionally subset the dataset
    if only_n_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(only_n_samples))  # pyright: ignore[reportAssignmentType, reportArgumentType]

    dataloader = DataLoader(dataset, batch_size=batch_size)  # pyright: ignore[reportArgumentType]

    data_rows = []

    samples_to_eval = len(dataset)
    pbar = tqdm(
        total=samples_to_eval,
        desc="Evaluating dataset samples",
    )

    for _batch_features, _batch_label in dataloader:
        batch_features = _batch_features.to(device)
        batch_label = _batch_label.to(device)

        # Initialize masks for the batch
        batch_masked_features = torch.zeros_like(batch_features).to(device)
        batch_feature_mask = torch.zeros_like(
            batch_features, dtype=torch.bool
        ).to(device)

        # Keep track of which samples still need processing
        active_indices = torch.arange(batch_features.shape[0], device=device)

        while len(active_indices) > 0:
            log.debug(f"Active indices in batch: {active_indices}")

            # Only choose features for active samples
            active_batch_selection = afa_select_fn(
                batch_masked_features[active_indices],
                batch_feature_mask[active_indices],
                batch_features[active_indices],
                batch_label[active_indices],
            ).squeeze(-1)
            assert active_batch_selection.ndim == 1, (
                f"batch_selection should be 1D, got {active_batch_selection.ndim}D with shape {active_batch_selection.shape}"
            )
            log.debug(f"Active batch selection: {active_batch_selection}")
            # Update masked features and feature mask individually
            # Stop action is handled later
            for i, active_idx in zip(
                range(active_batch_selection.shape[0]),
                active_indices,
                strict=True,
            ):
                sel = int(active_batch_selection[i].item())
                if sel > 0:
                    batch_feature_mask[active_idx, sel - 1] = True
                    batch_masked_features[active_idx, sel - 1] = (
                        batch_features[active_idx, sel - 1]
                    )
                # Log mask after update
                log.debug(
                    f"Sample {i} mask after update: {batch_feature_mask[i]}"
                )

            # Check which active samples are now finished, either due to early stopping or observing all features
            just_finished_indices = active_indices[
                (active_batch_selection == 0)
                | (batch_feature_mask[active_indices].all(dim=-1))
            ]
            log.debug(
                f"Just finished indices in batch: {just_finished_indices}"
            )

            if len(just_finished_indices) > 0:
                # Run predictions for just finished samples
                log.debug(
                    f"Running predictions for just finished samples {just_finished_indices}"
                )
                external_prediction = external_afa_predict_fn(
                    batch_masked_features[just_finished_indices],
                    batch_feature_mask[just_finished_indices],
                    batch_features[just_finished_indices],
                    batch_label[just_finished_indices],
                )
                external_prediction = torch.argmax(external_prediction, dim=-1)

                if builtin_afa_predict_fn is not None:
                    builtin_prediction = builtin_afa_predict_fn(
                        batch_masked_features[just_finished_indices],
                        batch_feature_mask[just_finished_indices],
                        batch_features[just_finished_indices],
                        batch_label[just_finished_indices],
                    )
                    builtin_prediction = torch.argmax(
                        builtin_prediction, dim=-1
                    )
                else:
                    builtin_prediction = None

                for i, idx in enumerate(just_finished_indices):
                    row = {
                        "features_chosen": batch_feature_mask[idx]
                        .sum()
                        .item(),
                        "predicted_label_external": external_prediction[
                            i
                        ].item(),
                        "true_label": batch_label[idx].argmax().item(),
                        "predicted_label_builtin": None
                        if builtin_prediction is None
                        else builtin_prediction[i].item(),
                        "acquisition_cost": (
                            acquisition_costs * batch_feature_mask[idx].float()
                        )
                        .sum()
                        .item(),
                    }
                    data_rows.append(row)
                pbar.update(len(just_finished_indices))
                pbar.refresh()

                # Remove finished samples from active indices
                active_indices = active_indices[
                    ~torch.isin(active_indices, just_finished_indices)
                ]

    df = pd.DataFrame(data_rows)

    return df
