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
    Structured-missingness initializer for CubeNMARDataset.

    It reveals archival hint features according to a hidden per-instance hint
    regime and blocks the relevant block in a subset of risky-context episodes.
    Both controls are encoded in hidden administrative coordinates that are
    always masked and never queryable.
    """

    def __init__(
        self,
        *,
        n_contexts: int,
        n_hint_features: int,
        block_size: int,
        n_safe_contexts: int,
        n_admin_features: int = 2,
    ):
        self.n_contexts = n_contexts
        self.n_hint_features = n_hint_features
        self.block_size = block_size
        self.n_safe_contexts = n_safe_contexts
        self.n_admin_features = n_admin_features
        self.seed: int | None = None
        self._last_forbidden_selection_mask: SelectionMask | None = None
        self._last_training_forbidden_mask: FeatureMask | None = None

    @override
    def set_seed(self, seed: int | None) -> None:
        self.seed = seed

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for CubeNMARInitializer"
        )

        flat_features = features.reshape(-1, feature_shape.numel())
        batch_size = flat_features.shape[0]
        context = flat_features[:, : self.n_contexts].argmax(dim=1)
        admin_start = self.n_contexts + self.n_hint_features
        hint_regime = flat_features[:, admin_start].round().long().clamp(0, 2)
        blocked_flag = flat_features[:, admin_start + 1] > 0.5

        observed = torch.zeros_like(flat_features, dtype=torch.bool)
        training_forbidden = torch.zeros_like(flat_features, dtype=torch.bool)
        forbidden_selection = torch.zeros(
            (
                batch_size,
                1
                + (
                    feature_shape.numel()
                    - self.n_contexts
                    - self.n_hint_features
                    - self.n_admin_features
                ),
            ),
            dtype=torch.bool,
            device=flat_features.device,
        )

        hint_start = self.n_contexts
        hint_end = hint_start + self.n_hint_features
        admin_end = admin_start + self.n_admin_features
        block_selection_start = 1
        block_feature_start = admin_end

        for sample_idx in range(batch_size):
            ctx = int(context[sample_idx].item())
            regime = int(hint_regime[sample_idx].item())

            if regime == 0:
                observed[sample_idx, hint_start + ctx] = True
            elif regime == 1:
                observed[sample_idx, hint_start + ctx] = True
                offset = 1 + (ctx % max(self.n_contexts - 1, 1))
                decoy = (ctx + offset) % self.n_contexts
                observed[sample_idx, hint_start + decoy] = True

            # Archival hints should remain visible if already observed, but
            # hidden hints and all admin controls are never acquirable.
            training_forbidden[sample_idx, hint_start:hint_end] = ~observed[
                sample_idx, hint_start:hint_end
            ]
            training_forbidden[sample_idx, admin_start:admin_end] = True

            is_risky = ctx >= self.n_safe_contexts
            is_blocked = is_risky and bool(blocked_flag[sample_idx].item())
            if is_blocked:
                block_start = block_selection_start + ctx * self.block_size
                forbidden_selection[
                    sample_idx, block_start : block_start + self.block_size
                ] = True
                feature_block_start = (
                    block_feature_start + ctx * self.block_size
                )
                training_forbidden[
                    sample_idx,
                    feature_block_start : feature_block_start
                    + self.block_size,
                ] = True

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

        # Fallback when initialize() was not called immediately beforehand.
        forbidden = torch.zeros_like(observed_mask, dtype=torch.bool)
        hint_start = self.n_contexts
        hint_end = hint_start + self.n_hint_features
        admin_end = hint_end + self.n_admin_features
        forbidden[..., hint_start:hint_end] = ~observed_mask[
            ..., hint_start:hint_end
        ]
        forbidden[..., hint_end:admin_end] = True
        return forbidden

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
