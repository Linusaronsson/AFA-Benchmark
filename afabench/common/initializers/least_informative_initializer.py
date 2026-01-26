from typing import final, override

import numpy as np
import torch
from sklearn.feature_selection import mutual_info_classif

from afabench.common.custom_types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
)


@final
class LeastInformativeInitializer(AFAInitializer):
    """
    Select features with LOWEST mutual information (adversarial baseline).

    Useful for robustness testing - how well do methods perform with poor initialization?
    """

    def __init__(self, num_initial_features: int):
        self.num_initial_features = num_initial_features
        self._cached_ranking: np.ndarray | None = None  # pyright: ignore[reportMissingTypeArgument]
        self._seed: int | None = None

    @override
    def set_seed(self, seed: int | None) -> None:
        self._seed = seed
        self._cached_ranking = None  # Clear cache if seed changes

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert features is not None, (
            f"{self.__class__.__name__} requires features"
        )
        assert label is not None, f"{self.__class__.__name__} requires label"
        assert feature_shape is not None, (
            "feature_shape must be provided for LeastInformativeInitializer"
        )

        num_features = feature_shape.numel()
        num_features_to_unmask = self.num_initial_features
        assert 0 <= num_features_to_unmask <= num_features, (
            f"num_initial_features={num_features_to_unmask} must be within "
            f"[0, {num_features}]"
        )

        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]

        # Flatten batch dimensions for sklearn
        features_flat = features.view(-1, *feature_shape)
        label_flat = label.view(-1)

        # Convert to numpy for mutual_info_classif
        train_features_np = features_flat.cpu().numpy()
        train_labels_np = label_flat.cpu().numpy()

        if num_features_to_unmask == 0:
            mask = torch.zeros(feature_shape, dtype=torch.bool)
            for _ in range(len(batch_shape)):
                mask = mask.unsqueeze(0)
            return mask.expand(batch_shape + feature_shape)

        # Compute MI once and cache ranking (NumPy array)
        if self._cached_ranking is None:
            # Reshape features to 2D for mutual_info_classif if necessary
            if train_features_np.ndim > 2:
                train_features_flat = train_features_np.reshape(
                    train_features_np.shape[0], -1
                )
            else:
                train_features_flat = train_features_np

            mi_scores = mutual_info_classif(
                train_features_flat, train_labels_np, random_state=self._seed
            )
            # Sort ascending by MI score (worst first)
            self._cached_ranking = np.argsort(mi_scores)

        selected_flat_indices_np = self._cached_ranking[
            :num_features_to_unmask
        ]

        # Convert to PyTorch tensor for unraveling
        selected_flat_indices = torch.tensor(
            selected_flat_indices_np, dtype=torch.long
        )

        # We can figure out the batch shape by subtracting the feature shape
        batch_shape = features.shape[: -len(feature_shape)]

        # Create mask
        mask = torch.zeros(feature_shape, dtype=torch.bool)
        multi_indices = torch.unravel_index(
            selected_flat_indices, feature_shape
        )
        mask[multi_indices] = True

        # Expand to match batch dimensions
        # Add singleton batch dimensions and expand
        for _ in range(len(batch_shape)):
            mask = mask.unsqueeze(0)
        return mask.expand(batch_shape + feature_shape)
