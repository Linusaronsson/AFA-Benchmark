from typing import final, override

import torch

from afabench.common.custom_types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
)


@final
class ZeroInitializer(AFAInitializer):
    """Cold-start initializer that selects no features."""

    def __init__(self):
        pass

    @override
    def set_seed(self, seed: int | None) -> None:
        # Zero initializer is deterministic, so seed has no effect
        pass

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for ZeroInitializer"
        )
        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]

        # Select no features (all False mask) and expand to batch dimensions
        mask = torch.zeros(feature_shape, dtype=torch.bool)
        # Add singleton batch dimensions and expand
        for _ in range(len(batch_shape)):
            mask = mask.unsqueeze(0)
        return mask.expand(batch_shape + feature_shape)
