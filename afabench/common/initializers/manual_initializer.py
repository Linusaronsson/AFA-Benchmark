from typing import final, override

import torch

from afabench.common.custom_types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
)


@final
class ManualInitializer(AFAInitializer):
    """Use explicitly specified feature indices."""

    def __init__(self, flat_feature_indices: list[int]):
        self.flat_feature_indices = torch.tensor(
            flat_feature_indices, dtype=torch.long
        )

    @override
    def set_seed(self, seed: int | None) -> None:
        # Manual initializer is deterministic, so seed has no effect
        pass

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for ManualInitializer"
        )
        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]
        num_features_to_unmask = self.flat_feature_indices.numel()
        num_features = feature_shape.numel()

        # Ensure feature indices are valid
        assert (self.flat_feature_indices >= 0).all() and (  # noqa: PT018
            self.flat_feature_indices < num_features
        ).all(), "Feature index out of bounds"

        mask = torch.zeros(feature_shape, dtype=torch.bool)
        multi_indices = torch.unravel_index(
            self.flat_feature_indices, feature_shape
        )
        mask[multi_indices] = True

        assert mask.sum() == num_features_to_unmask, f"Expected {
            num_features_to_unmask
        } features, got {mask.sum()}"

        # Expand to match batch dimensions
        # Add singleton batch dimensions and expand
        for _ in range(len(batch_shape)):
            mask = mask.unsqueeze(0)
        return mask.expand(batch_shape + feature_shape)

    @override
    def __repr__(self) -> str:
        return (
            f"ManualInitializer(features={self.flat_feature_indices.tolist()})"
        )
