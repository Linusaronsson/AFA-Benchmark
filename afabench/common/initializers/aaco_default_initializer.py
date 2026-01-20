from typing import ClassVar, final, override

import torch

from afabench.common.custom_types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
)


@final
class AACODefaultInitializer(AFAInitializer):
    """
    AACO paper's deterministic initial feature selection.

    Uses dataset-specific initial features as per the original paper:
    - cube: feature 6
    - mnist/fashionmnist: feature 100
    - others: middle feature

    # 2026-01-20 TEST: Use random first feature for ablation
    """

    DATASET_INITIAL_FEATURES: ClassVar = {
        "cube": 6,
        "cubesimple": 3,
        "grid": 1,
        "gas10": 6,
        "mnist": 100,
        "fashionmnist": 100,
        "syntheticmnist": 100,
        "afacontext": 0,  # context feature first
    }

    def __init__(self, dataset_name: str):
        normalized = dataset_name.lower()
        normalized = normalized.replace("_", "")
        normalized = normalized.replace("withoutnoise", "")
        self.dataset_name = normalized

    @override
    def set_seed(self, seed: int | None) -> None:
        # AACODefaultInitializer is deterministic, so seed has no effect
        pass

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for AACODefaultInitializer"
        )
        num_features = feature_shape.numel()

        # initial_feature_flat_index = self.DATASET_INITIAL_FEATURES.get(
        #     self.dataset_name,
        #     num_features // 2,  # fallback: middle feature
        # )

        # 2026-01-20 TEST: Use random first feature for ablation
        # Make sure we have same type as above for consistency
        initial_feature_flat_index = torch.randint(
            low=0, high=num_features, size=(1,)
        ).item()
        assert isinstance(initial_feature_flat_index, int), (
            "Random initial feature index should be an integer"
        )

        # AACO always starts with exactly 1 feature
        # This implicitly means the intended unmask_ratio leads to 1 feature
        # The ManualInitializer has a similar check for consistency.

        # Ensure the selected feature is within bounds
        assert 0 <= initial_feature_flat_index < num_features, (
            f"Initial feature index {initial_feature_flat_index} for dataset "
            f"'{self.dataset_name}' is out of bounds for {
                num_features
            } features."
        )

        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]

        mask = torch.zeros(feature_shape, dtype=torch.bool)
        initial_feature_flat_index_tensor = torch.tensor(
            initial_feature_flat_index, dtype=torch.long
        )
        multi_indices = torch.unravel_index(
            initial_feature_flat_index_tensor, feature_shape
        )
        # multi_indices will be a tuple of arrays, e.g., (array([x]), array([y]), ...).
        # We need to ensure it's applied correctly for scalar indices too by directly passing the tuple.
        mask[multi_indices] = True

        # Expand to match batch dimensions
        # Add singleton batch dimensions and expand
        for _ in range(len(batch_shape)):
            mask = mask.unsqueeze(0)
        return mask.expand(batch_shape + feature_shape)
