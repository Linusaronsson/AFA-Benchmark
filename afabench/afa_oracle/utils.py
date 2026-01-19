"""Shared utilities for AACO methods."""

import torch


def compute_patch_selection_mask(
    feature_mask: torch.Tensor,
    selection_size: int,
    feature_shape: torch.Size,
) -> torch.Tensor:
    """
    Convert feature-level mask to patch-level selection mask.

    Args:
        feature_mask: Feature mask with shape (*batch, *feature_shape) or
            (*batch, n_features) where n_features = prod(feature_shape).
        selection_size: Number of patches (must be a perfect square)
        feature_shape: Shape of features (H, W) or (C, H, W)

    Returns:
        Patch selection mask with shape (batch_size, selection_size)
    """
    if len(feature_shape) == 3:
        channels, height, width = feature_shape
    else:
        channels = 1
        height, width = feature_shape

    mask_width = int(selection_size**0.5)
    assert mask_width * mask_width == selection_size, (
        f"Patch selection size must be a square number, got {selection_size}."
    )
    assert height % mask_width == 0, (
        f"Patch grid must evenly divide image height, got "
        f"height={height}, mask_width={mask_width}."
    )
    assert width % mask_width == 0, (
        f"Patch grid must evenly divide image width, got "
        f"width={width}, mask_width={mask_width}."
    )

    patch_h = height // mask_width
    patch_w = width // mask_width

    n_features = feature_shape.numel()
    mask = feature_mask
    assert (
        mask.shape[-len(feature_shape) :] == feature_shape
        or mask.shape[-1] == n_features
    ), (
        f"feature_mask shape {tuple(mask.shape)} does not match "
        f"feature_shape {tuple(feature_shape)}."
    )
    if mask.shape[-len(feature_shape) :] == feature_shape:
        mask = mask.reshape(-1, *feature_shape)
    else:
        mask = mask.reshape(-1, n_features).reshape(-1, *feature_shape)

    if len(feature_shape) == 2:
        mask = mask.unsqueeze(1)

    batch_size = mask.shape[0]
    mask = mask.view(
        batch_size,
        channels,
        mask_width,
        patch_h,
        mask_width,
        patch_w,
    )
    # A patch is selected if ANY pixel in ANY channel is observed
    return mask.any(dim=(1, 3, 5)).view(batch_size, selection_size)


def count_acquisitions(
    feature_mask: torch.Tensor,
    feature_shape: torch.Size | None,
    selection_size: int | None,
) -> int:
    """
    Count acquisitions in selection space (patches or features).

    Args:
        feature_mask: Boolean mask of observed features (flattened)
        feature_shape: Shape of features (for patch-based counting)
        selection_size: Number of selections (patches). If None or equal to
                       feature count, counts individual features.

    Returns:
        Number of acquisitions made
    """
    n_features = (
        feature_shape.numel()
        if feature_shape is not None
        else feature_mask.shape[-1]
    )
    use_patches = (
        selection_size is not None
        and selection_size < n_features
        and feature_shape is not None
        and len(feature_shape) in (2, 3)
    )

    if use_patches:
        assert selection_size is not None
        assert feature_shape is not None
        patch_mask = compute_patch_selection_mask(
            feature_mask, selection_size, feature_shape
        )
        return int(patch_mask.sum().item())
    return int(feature_mask.bool().sum().item())


def flatten_for_aaco(
    features: torch.Tensor,
    feature_mask: torch.Tensor,
    feature_shape: torch.Size | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Size]:
    """
    Flatten features and mask for AACO processing.

    Args:
        features: Features tensor with shape (*batch, *feature_shape)
        feature_mask: Mask tensor with shape (*batch, *feature_shape)
        feature_shape: Shape of features excluding batch dimensions

    Returns:
        Tuple of (flattened_features, flattened_mask, batch_shape)
    """
    if feature_shape is None:
        batch_shape = feature_mask.shape[:-1]
        n_features = features.shape[-1]
    else:
        batch_shape = feature_mask.shape[: -len(feature_shape)]
        n_features = feature_shape.numel()

    features_flat = features.view(-1, n_features)
    mask_flat = feature_mask.view(-1, n_features)

    return features_flat, mask_flat, batch_shape
