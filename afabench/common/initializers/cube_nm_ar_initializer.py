from typing import final, override

import torch

from afabench.common.custom_types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
    SelectionMask,
)


@final
class CubeNMARInitializer(AFAInitializer):
    """
    Risk-aware training-support initializer for CubeNMARDataset.

    Training-time support restriction is injected here: for a subset of
    risky-context samples, the rescue feature is unavailable during training
    even though it is queryable again at evaluation.
    """

    def __init__(
        self,
        *,
        n_contexts: int,
        block_size: int,
        n_safe_contexts: int,
        risky_rescue_missing_probability: float = 0.3,
    ):
        if not (0.0 <= risky_rescue_missing_probability <= 1.0):
            msg = (
                "Expected risky_rescue_missing_probability in [0, 1], got "
                f"{risky_rescue_missing_probability=}."
            )
            raise ValueError(msg)

        self.n_contexts = n_contexts
        self.block_size = block_size
        self.n_safe_contexts = n_safe_contexts
        self.risky_rescue_missing_probability = (
            risky_rescue_missing_probability
        )
        self.seed: int | None = None
        self._rng: torch.Generator | None = None
        self._last_forbidden_selection_mask: SelectionMask | None = None
        self._last_training_forbidden_mask: FeatureMask | None = None

    def _get_rng(self) -> torch.Generator:
        if self._rng is None:
            self._rng = torch.Generator()
            if self.seed is not None:
                self._rng.manual_seed(self.seed)
        return self._rng

    @override
    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self._rng = None

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        del label
        assert feature_shape is not None, (
            "feature_shape must be provided for CubeNMARInitializer"
        )

        flat_features = features.reshape(-1, feature_shape.numel())
        batch_size = flat_features.shape[0]
        context = flat_features[:, : self.n_contexts].argmax(dim=1)
        rescue_feature_idx = feature_shape.numel() - 1
        rescue_selection_idx = 1 + self.n_contexts * self.block_size

        observed = torch.zeros_like(flat_features, dtype=torch.bool)
        training_forbidden = torch.zeros_like(flat_features, dtype=torch.bool)
        forbidden_selection = torch.zeros(
            (batch_size, 1 + feature_shape.numel() - self.n_contexts),
            dtype=torch.bool,
            device=flat_features.device,
        )

        risky_indices = (
            (context >= self.n_safe_contexts)
            .nonzero(as_tuple=False)
            .squeeze(1)
        )
        if risky_indices.numel() > 0:
            risky_draws = torch.rand(
                (int(risky_indices.numel()),),
                generator=self._get_rng(),
            ).to(flat_features.device)
            missing_mask = risky_draws < self.risky_rescue_missing_probability
            missing_indices = risky_indices[missing_mask]
            if missing_indices.numel() > 0:
                training_forbidden[missing_indices, rescue_feature_idx] = True
                forbidden_selection[missing_indices, rescue_selection_idx] = (
                    True
                )

        self._last_forbidden_selection_mask = forbidden_selection.reshape(
            *features.shape[: -len(feature_shape)],
            forbidden_selection.shape[-1],
        )
        self._last_training_forbidden_mask = training_forbidden.reshape(
            features.shape[: -len(feature_shape)] + feature_shape
        )
        return observed.reshape(
            features.shape[: -len(feature_shape)] + feature_shape
        )

    @override
    def get_training_forbidden_mask(
        self,
        observed_mask: FeatureMask,
    ) -> FeatureMask:
        if self._last_training_forbidden_mask is not None:
            return self._last_training_forbidden_mask
        return torch.zeros_like(observed_mask, dtype=torch.bool)

    def get_forbidden_selection_mask(
        self,
        _observed_mask: FeatureMask,
        _feature_shape: torch.Size,
    ) -> SelectionMask:
        if self._last_forbidden_selection_mask is None:
            msg = (
                "CubeNMARInitializer has no cached forbidden selection mask. "
                "Call initialize() before requesting forbidden selections."
            )
            raise RuntimeError(msg)
        return self._last_forbidden_selection_mask
