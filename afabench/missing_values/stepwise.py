"""Shared state transition for stepwise feature restoration."""

import torch

from afabench.components.methods.rl.common.custom_types import (
    AFAFeatureRestorationFn,
)


def restore_acquired_features(
    features: torch.Tensor,
    feature_mask: torch.Tensor,
    new_feature_mask: torch.Tensor,
    source_availability: torch.Tensor,
    restoration_fn: AFAFeatureRestorationFn | None,
) -> torch.Tensor:
    """Restore newly acquired missing values without exposing hidden values."""
    if restoration_fn is None:
        return features
    newly_revealed = new_feature_mask & ~feature_mask
    restore_mask = newly_revealed & ~source_availability
    restore_rows = restore_mask.flatten(start_dim=1).any(dim=1)
    if not restore_rows.any():
        return features

    conditioning_mask = feature_mask | (newly_revealed & source_availability)
    conditioning_features = features.clone()
    conditioning_features[~conditioning_mask] = 0.0
    estimates = restoration_fn(
        conditioning_features[restore_rows],
        conditioning_mask[restore_rows],
    )
    restored = features.clone()
    row_values = restored[restore_rows]
    row_restore_mask = restore_mask[restore_rows]
    row_values[row_restore_mask] = estimates[row_restore_mask]
    restored[restore_rows] = row_values
    return restored
