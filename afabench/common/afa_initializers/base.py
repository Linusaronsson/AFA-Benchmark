from abc import ABC, abstractmethod

import numpy as np
import torch

from afabench.common.custom_types import (
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)


class AFAInitializer(ABC):
    """
    Abstract base class for selecting initial features in warm-start scenarios.

    Subclasses must implement select_features() to define feature selection logic.
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize the warm-start strategy.

        Args:
            seed: Random seed for reproducibility. If None, results may vary.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def select_features(
        self,
        n_features_total: int,
        n_features_select: int,
        train_features: Features | None = None,
        train_labels: Label | None = None,
    ) -> list[int]:
        """
        Select which features to provide as warm-start.

        Args:
            n_features_total: Total number of features available
            n_features_select: Number of features to select for warm-start
            train_features: Training features (shape: [n_samples, n_features])
                           Required for data-dependent strategies
            train_labels: Training labels (shape: [n_samples,])
                         Required for data-dependent strategies

        Returns:
            List of feature indices to warm-start with (length = n_features_select)

        Raises:
            ValueError: If required data is not provided for data-dependent strategies
        """

    def initialize(
        self,
        features: Features,
        n_features_select: int = 0,
        train_features: Features | None = None,
        train_labels: Label | None = None,
    ) -> tuple[MaskedFeatures, FeatureMask]:
        """
        Create initial masked features and feature mask.

        Args:
            features: Input features (batch_size, n_features)
            n_features_select: How many features to reveal initially
            train_features: Training features for data-dependent strategies
            train_labels: Training labels for data-dependent strategies

        Returns:
            (masked_features, feature_mask) tuple ready for AFA loop
        """
        n_features_total = features.shape[-1]

        # Get indices to reveal
        if n_features_select > 0:
            indices = self.select_features(
                n_features_total=n_features_total,
                n_features_select=n_features_select,
                train_features=train_features,
                train_labels=train_labels,
            )
        else:
            indices = []

        # Build mask (batch_size, n_features)
        feature_mask = torch.zeros(
            features.shape, dtype=torch.bool, device=features.device
        )
        for idx in indices:
            feature_mask[..., idx] = True

        # Apply mask
        masked_features = features.clone()
        masked_features[~feature_mask] = 0.0

        return masked_features, feature_mask

    def __repr__(self) -> str:
        """String representation for logging."""
        return f"{self.__class__.__name__}(seed={self.seed})"
