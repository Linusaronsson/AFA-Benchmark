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
class AFAContextUnmasker(AFAUnmasker):
    """The first selection unmasks **all** the context features, otherwise this unmasker behaves like DirectUnmasker."""

    def __init__(self, n_contexts: int):
        self.n_contexts = n_contexts

    @override
    def set_seed(self, seed: int | None) -> None:
        # This unmasker is deterministic
        pass

    @override
    def get_selection_costs(self, feature_costs: torch.Tensor) -> torch.Tensor:
        # TODO
        return feature_costs

    @override
    def get_n_selections(self, feature_shape: torch.Size) -> int:
        # TODO
        return feature_shape.numel()

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
        # TODO
        assert feature_shape is not None, "feature_shape must be provided"
        assert afa_selection.shape[-1] == 1, (
            "AFASelection must have shape [..., 1]"
        )

        # Get batch shape by removing the last dimension (which is 1)
        batch_shape = afa_selection.shape[:-1]

        # Flatten batch dimensions for processing
        afa_selection_flat = afa_selection.view(-1, 1)
        feature_mask_flat = feature_mask.view(-1, *feature_shape)

        feature_indices = afa_selection_flat.squeeze(1)

        new_feature_mask_flat = feature_mask_flat.clone()

        # Only unmask features where selection >= 0 (selection < 0 is ignored)
        valid_selections = afa_selection_flat.squeeze(1) >= 0
        if valid_selections.any():
            valid_feature_indices = feature_indices[valid_selections]
            valid_batch_indices = torch.arange(
                feature_mask_flat.size(0), device=feature_mask.device
            )[valid_selections]

            # Convert feature_indices to n-dimensional indices using unravel_index
            multi_indices = torch.unravel_index(
                valid_feature_indices, feature_shape
            )
            new_feature_mask_flat[valid_batch_indices, *multi_indices] = True

        # Reshape back to original batch shape + feature shape
        return new_feature_mask_flat.view(batch_shape + feature_shape)
