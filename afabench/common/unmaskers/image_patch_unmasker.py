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
        self.selection_size = self.low_dim_image_side_length**2

    def _selections_to_lowres_image_mask(
        self, selections: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert selections to low-resolution image masks.

        Args:
            selections: Tensor of shape (batch, 1) containing selection indices

        Returns:
            Low-res mask of shape (batch, h_low, w_low)
        """
        selections_1d = F.one_hot(selections, num_classes=self.selection_size)
        selections_lowres_image = selections_1d.view(
            selections.shape[0],
            self.low_dim_image_side_length,
            self.low_dim_image_side_length,
        )
        return selections_lowres_image

    def _upscale_lowres_image_mask(
        self, lowres_image_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Upscale a low-resolution image mask to high-resolution image mask.

        Args:
            lowres_image_mask: Low-res mask of shape (batch, h_low, w_low) or (h_low, w_low)

        Returns:
            High-res mask of shape (batch, h, w)
        """
        # Add channel dimension if not present
        if lowres_image_mask.dim() == 2:
            lowres_image_mask = lowres_image_mask.unsqueeze(0)

        # Add batch dimension if not present
        if lowres_image_mask.dim() == 3:
            lowres_image_mask = lowres_image_mask.unsqueeze(0)

        # Upscale to highres image
        highres_image_mask = F.interpolate(
            lowres_image_mask.float(),
            scale_factor=self.patch_size,
            mode="nearest-exact",
        ).bool()

        return highres_image_mask

    def _get_image_mask_from_selection(self, selection: int) -> torch.Tensor:
        selection_tensor = torch.tensor([[selection]])
        lowres_image_mask = self._selections_to_lowres_image_mask(
            selection_tensor
        )
        # Upscale to highres image
        highres_image_mask = self._upscale_lowres_image_mask(lowres_image_mask)

        # Remove batch and channel dimensions before returning
        return highres_image_mask.squeeze(0).squeeze(0)

    @override
    def get_selection_costs(self, feature_costs: torch.Tensor) -> torch.Tensor:
        """Sum the feature cost within each patch."""
        assert feature_costs.shape == (
            self.n_channels,
            self.image_side_length,
            self.image_side_length,
        )

        selection_costs = torch.zeros(self.selection_size)
        for selection_idx in range(self.selection_size):
            image_mask = self._get_image_mask_from_selection(selection_idx)
            feature_cost_in_patch = (feature_costs * image_mask).sum()
            selection_costs[selection_idx] = feature_cost_in_patch
        return selection_costs

    @override
    def get_n_selections(self, feature_shape: torch.Size) -> int:
        # The number of selections is equal to the number of patches
        return self.selection_size

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

        # Convert afa selection into low-dimensional image mask
        sel = afa_selection.view(-1, 1).to(torch.long)
        lowres_image_mask = self._selections_to_lowres_image_mask(sel)

        # Add channel dimension and expand to all channels
        lowres_image_mask = lowres_image_mask.unsqueeze(1).expand(
            -1, self.n_channels, -1, -1
        )

        # Upscale to high-resolution image mask
        highres_image_mask = self._upscale_lowres_image_mask(lowres_image_mask)

        # Flatten batch dimensions for processing
        feature_mask_flat = feature_mask.view(
            batch_size,
            *feature_mask.shape[-len(highres_image_mask.shape[-3:]) :],
        )

        # Convert image mask to feature mask and add to previous feature mask
        new_feature_mask_flat = feature_mask_flat | highres_image_mask

        # Reshape back to original batch shape + feature shape
        return new_feature_mask_flat.view(
            batch_shape
            + feature_mask.shape[-len(highres_image_mask.shape[-3:]) :]
        )
