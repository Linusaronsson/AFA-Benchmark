"""
Generative restoration of training-missing features.

Shared helpers for episode-start restoration: derive the per-instance
training support mask from an initializer, load a pretrained PVAE, and
complete masked rows with a single joint reconstruction per row.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from afabench.common.bundle import load_bundle

if TYPE_CHECKING:
    from pathlib import Path

    from afabench.afa_rl.zannone2019.models import (
        Zannone2019PretrainingModel,
    )
    from afabench.common.custom_types import AFAInitializer
    from afabench.common.torch_bundle import TorchModelBundle


def derive_train_support_masks(
    initializer: AFAInitializer,
    *,
    seed: int,
    features: torch.Tensor,
    feature_shape: torch.Size,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    initializer.set_seed(seed)
    observed_mask = initializer.initialize(
        features=features,
        feature_shape=feature_shape,
    ).bool()
    forbidden_mask = initializer.get_training_forbidden_mask(
        observed_mask
    ).bool()
    if bool((observed_mask & forbidden_mask).any().item()):
        msg = (
            "Initializer returned overlapping observed and forbidden "
            "training masks."
        )
        raise ValueError(msg)
    train_support_mask = ~forbidden_mask
    return observed_mask, train_support_mask


def load_pvae_model(
    pretrained_model_bundle_path: Path,
    *,
    device: torch.device,
) -> Zannone2019PretrainingModel:
    pretrained_model, _ = load_bundle(
        pretrained_model_bundle_path,
        device=device,
    )
    torch_model_bundle = cast(
        "TorchModelBundle",
        cast("object", pretrained_model),
    )
    pvae_model = cast(
        "Zannone2019PretrainingModel",
        torch_model_bundle.model,
    )
    pvae_model = pvae_model.to(device)
    pvae_model.eval()
    return pvae_model


def restore_missing_features_with_pvae(
    x: torch.Tensor,
    train_support_mask: torch.BoolTensor,
    *,
    pvae_model: Zannone2019PretrainingModel,
    hide_val: float = 0.0,
    label: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Complete unsupported entries of `x` with PVAE reconstructions.

    Each row is completed with one joint reconstruction (a single latent
    sample decodes all missing entries together). If `label` is given
    (one-hot, shape (batch, n_classes)), the reconstruction is
    class-conditional; supported entries are always kept factual.
    """
    missing = ~train_support_mask
    masked_x = x.clone()
    masked_x[missing] = hide_val

    if label is not None:
        label = label.to(device=masked_x.device, dtype=torch.float32)

    with torch.no_grad():
        _z, reconstructed = pvae_model.masked_reconstruction(
            masked_features=masked_x,
            feature_mask=train_support_mask,
            n_classes=pvae_model.n_classes,
            label=label,
        )

    restored_x = x.clone()
    restored_x[missing] = reconstructed.to(restored_x.dtype)[missing]
    return restored_x
