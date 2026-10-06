import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from afabench.components.methods.discriminative.common.models import (
    GreedyAFAClassifier,
    MaskingPretrainer,
    Predictor,
    ResNet18Backbone,
    resnet18,
    resnet50,
)
from afabench.components.methods.discriminative.common.utils import (
    MaskLayer2d,
)
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSImageArchitectureConfig,
    GDFSPretrainingConfig,
)
from afabench.training.inputs import TrainingInputs

log = logging.getLogger(__name__)


def pretrain_image(
    cfg: GDFSPretrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: TrainingInputs,
) -> GreedyAFAClassifier:
    log.debug(cfg)
    assert isinstance(cfg.architecture, GDFSImageArchitectureConfig)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)
    train_dataset = inputs.train_dataset()
    d_out = train_dataset.label_shape[0]
    val_dataset = inputs.val_dataset()
    train_loader = DataLoader(
        train_dataset,  # pyright: ignore[reportArgumentType]
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,  # pyright: ignore[reportArgumentType]
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    backbone_type = cfg.architecture.backbone_type
    if backbone_type == "resnet18":
        base = resnet18(pretrained=True)
    elif backbone_type == "resnet50":
        base = resnet50(pretrained=True)
    else:
        msg = f"Unsupported backbone type: {backbone_type}"
        raise ValueError(msg)
    backbone, expansion = ResNet18Backbone(base)
    predictor = Predictor(backbone, expansion, num_classes=d_out).to(device)
    image_size = cfg.architecture.image_size
    patch_size = cfg.architecture.patch_size
    assert image_size % patch_size == 0, (
        "image_size must be divisible by patch_size"
    )
    mask_width = image_size // patch_size
    architecture: dict[str, Any] = {
        "type": backbone_type,
        "backbone": backbone_type,
        "image_size": image_size,
        "patch_size": patch_size,
        "mask_width": mask_width,
        "d_out": d_out,
    }
    mask_layer = MaskLayer2d(
        mask_width=mask_width, patch_size=patch_size, append=False
    )
    pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
    pretrain.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        loss_fn=nn.CrossEntropyLoss(),
        patience=cfg.patience,
        verbose=True,
        min_mask=cfg.min_masking_probability,
        max_mask=cfg.max_masking_probability,
        metric_logger=metric_logger,
        metric_prefix="gdfs_pretrain",
    )
    bundle_obj = GreedyAFAClassifier(
        predictor=predictor,
        architecture=architecture,
        device=torch.device("cpu"),
    )
    return bundle_obj
