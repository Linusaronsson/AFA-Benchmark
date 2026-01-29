import gc
import logging
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from afabench.afa_discriminative.models import (
    GreedyAFAClassifier,
    MaskingPretrainer,
    Predictor,
    ResNet18Backbone,
    resnet18,
)
from afabench.afa_discriminative.utils import MaskLayer2d
from afabench.common.bundle import (
    load_bundle,
    save_bundle,
)
from afabench.common.config_classes import Gadgil2023Pretraining2DConfig
from afabench.common.custom_types import AFADataset  # noqa: TC001
from afabench.common.utils import set_seed

log = logging.getLogger(__name__)


def pretrain_image(cfg: Gadgil2023Pretraining2DConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    train_dataset, _ = load_bundle(Path(cfg.train_dataset_bundle_path))
    train_dataset = cast("AFADataset", cast("object", train_dataset))
    d_out = train_dataset.label_shape[0]
    val_dataset, _ = load_bundle(Path(cfg.val_dataset_bundle_path))
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

    base = resnet18(pretrained=True)
    backbone, expansion = ResNet18Backbone(base)
    predictor = Predictor(backbone, expansion, num_classes=d_out).to(device)

    image_size = cfg.image_size
    patch_size = cfg.patch_size
    assert image_size % patch_size == 0, (
        "image_size must be divisible by patch_size"
    )
    mask_width = image_size // patch_size
    architecture: dict[str, Any] = {
        "type": "resnet18",
        "backbone": "resnet18",
        "image_size": image_size,
        "patch_size": patch_size,
        "mask_width": mask_width,
        "d_out": d_out,
    }

    mask_layer = MaskLayer2d(
        mask_width=mask_width, patch_size=patch_size, append=False
    )
    print("Pretraining predictor")
    print("-" * 8)
    pretrain = MaskingPretrainer(predictor, mask_layer).to(device)
    pretrain.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        nepochs=cfg.nepochs,
        loss_fn=nn.CrossEntropyLoss(),
        val_loss_fn=Accuracy(task="multiclass", num_classes=d_out).to(device),
        val_loss_mode="max",
        patience=cfg.patience,
        verbose=True,
        min_mask=cfg.min_masking_probability,
        max_mask=cfg.max_masking_probability,
    )

    metadata = {
        "model_type": "Gadgil2023Classifier",
        "pretrain_config": OmegaConf.to_container(cfg),
    }
    bundle_obj = GreedyAFAClassifier(
        predictor=predictor,
        architecture=architecture,
        device=torch.device("cpu"),
    )

    save_bundle(
        obj=bundle_obj,
        path=Path(cfg.save_path),
        metadata=metadata,
    )

    log.info(f"Gadgil2023 pretrained model saved to: {cfg.save_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
