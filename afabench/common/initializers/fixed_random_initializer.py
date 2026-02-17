from typing import final, override

import torch

from afabench.common.custom_types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
)


@final
class FixedRandomInitializer(AFAInitializer):
    """Select random features once and reuse them across all episodes."""

    def __init__(self, num_initial_features: int):
        self.num_initial_features = num_initial_features
        self._cached_mask: FeatureMask | None = None
        self.rng = torch.Generator()

    @override
    def set_seed(self, seed: int | None) -> None:
        if seed is not None:
            self.rng.manual_seed(seed)
        else:
            # Re-initialize with a new random seed if None is provided
            self.rng = torch.Generator()

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert (
            feature_shape is not None
        ), "feature_shape must be provided for FixedRandomInitializer"
        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]
        # Select once and cache
        if self._cached_mask is None:
            assert (
                feature_shape is not None
            ), "feature_shape must be provided for FixedRandomInitializer"
            num_features = feature_shape.numel()
            num_features_to_unmask = self.num_initial_features
            assert 0 <= num_features_to_unmask <= num_features, (
                f"num_initial_features={num_features_to_unmask} must be "
                f"within [0, {num_features}]"
            )
            # Sample feature indices
            flat_indices = torch.randperm(num_features, generator=self.rng)[
                :num_features_to_unmask
            ]
            # Unflatten
            multi_indices = torch.unravel_index(flat_indices, feature_shape)
            # Turn into mask
            self._cached_mask = torch.zeros(feature_shape, dtype=torch.bool)
            self._cached_mask[multi_indices] = True

        assert self._cached_mask is not None
        # Expand the mask to match batch dimensions
        # Create a copy and add singleton batch dimensions
        expanded_mask = self._cached_mask
        for _ in range(len(batch_shape)):
            expanded_mask = expanded_mask.unsqueeze(0)
        return expanded_mask.expand(batch_shape + feature_shape)
