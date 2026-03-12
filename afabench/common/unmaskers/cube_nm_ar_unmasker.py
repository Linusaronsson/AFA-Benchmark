from typing import final, override

import torch

from afabench.common.custom_types import (
    AFASelection,
    AFAUnmasker,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


@final
class CubeNMARUnmasker(AFAUnmasker):
    """Group cheap context features and expose all remaining features directly."""

    def __init__(self, n_contexts: int):
        self.n_contexts = n_contexts

    @override
    def set_seed(self, seed: int | None) -> None:
        pass

    @override
    def get_selection_costs(self, feature_costs: torch.Tensor) -> torch.Tensor:
        assert feature_costs.ndim == 1
        selection_costs = torch.full(
            (self.get_n_selections(feature_costs.shape),), float("nan")
        )
        selection_costs[0] = feature_costs[: self.n_contexts].sum()
        selection_costs[1:] = feature_costs[self.n_contexts :]
        return selection_costs

    @override
    def get_n_selections(self, feature_shape: torch.Size) -> int:
        assert len(feature_shape) == 1
        n_features = feature_shape.numel()
        return 1 + (n_features - self.n_contexts)

    @override
    def unmask(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        afa_selection: AFASelection,
        selection_mask: SelectionMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert masked_features.ndim == 2, (
            "Expected a single batch dimension and a single feature dimension."
        )
        new_feature_mask = feature_mask.clone()

        for sample_idx in range(masked_features.shape[0]):
            selection_idx = int(afa_selection[sample_idx])
            if selection_idx == 0:
                new_feature_mask[sample_idx, : self.n_contexts] = True
            else:
                new_feature_mask[
                    sample_idx, self.n_contexts + (selection_idx - 1)
                ] = True

        return new_feature_mask
