from typing import final, override

import torch
from torch.nn import functional as F

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
class ImagePatchUnmasker(AFAUnmasker):
    def __init__(
        self, image_side_length: int, patch_size: int, n_channels: int
    ):
        assert image_side_length % patch_size == 0, (
            "Image side length must be divisible by patch size"
        )
        self.image_side_length = image_side_length
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.low_dim_image_side_length = image_side_length // patch_size
        self.afa_selection_size = self.low_dim_image_side_length**2

    @override
    def get_n_selections(self, feature_shape: torch.Size) -> int:
        # The number of selections is equal to the number of patches
        return self.afa_selection_size

    @override
    def set_seed(self, seed: int | None) -> None:
        # This unmasker is deterministic
        pass

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
        assert afa_selection.shape[-1] == 1, (
            "AFASelection must have shape [..., 1]"
        )

        # Get batch shape by removing the last dimension (which is 1)
        batch_shape = afa_selection.shape[:-1]
        batch_size = int(torch.prod(torch.tensor(batch_shape)))

        # Convert afa selection into 1D mask
        sel = afa_selection.view(-1, 1).to(torch.long)
        afa_selection_1d = F.one_hot(sel, num_classes=self.afa_selection_size)

        # Convert to low-dimensional image mask
        afa_selection_low_dim_image = afa_selection_1d.view(
            batch_size,
            self.low_dim_image_side_length,
            self.low_dim_image_side_length,
        )

        # Add batch dimension and channel dimension, expand channels
        afa_selection_low_dim_image = afa_selection_low_dim_image.unsqueeze(
            1
        ).expand(-1, self.n_channels, -1, -1)

        # Upscale image mask, converting between float and bool
        afa_selection_image = F.interpolate(
            afa_selection_low_dim_image.float(),
            scale_factor=self.patch_size,
            mode="nearest-exact",
        ).bool()

        # Flatten batch dimensions for processing
        feature_mask_flat = feature_mask.view(
            batch_size,
            *feature_mask.shape[-len(afa_selection_image.shape[-3:]) :],
        )

        # Convert image mask to feature mask and add to previous feature mask
        new_feature_mask_flat = feature_mask_flat | afa_selection_image

        # Reshape back to original batch shape + feature shape
        return new_feature_mask_flat.view(
            batch_shape
            + feature_mask.shape[-len(afa_selection_image.shape[-3:]) :]
        )
