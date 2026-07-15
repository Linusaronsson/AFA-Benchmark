import gc
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from afabench.components.methods.discriminative.common.models import (
    ConvNet,
    GreedyAFAClassifier,
    ResNet18Backbone,
    resnet18,
    resnet50,
)
from afabench.components.methods.discriminative.common.utils import (
    MaskLayer2d,
    afa_discriminative_training_prep,
)
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
    GreedyDynamicSelection,
)
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSImageArchitectureConfig,
    GDFSTrainingConfig,
)
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.core.utils import set_seed
from afabench.training.smoke_test import dataset_subset, training_batch_size

log = logging.getLogger(__name__)


def _load_backbone(backbone_type: str) -> tuple[torch.nn.Module, int]:
    if backbone_type == "resnet18":
        base = resnet18(pretrained=True)
    elif backbone_type == "resnet50":
        base = resnet50(pretrained=True)
    else:
        msg = f"Unsupported backbone type: {backbone_type}"
        raise ValueError(msg)
    return ResNet18Backbone(base)


def train_image(
    cfg: GDFSTrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
) -> None:
    log.debug(cfg)
    assert isinstance(cfg.architecture, GDFSImageArchitectureConfig)
    assert cfg.device is not None, "device must be configured"
    assert cfg.hard_budget is not None, "hard_budget must be configured"
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)
    if cfg.smoke_test:
        cfg.nepochs = 1
        cfg.patience = 1
    train_dataset, val_dataset, initializer, unmasker, _ = (
        afa_discriminative_training_prep(
            train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
            val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
            initializer_cfg=cfg.initializer,
            unmasker_cfg=cfg.unmasker,
        )
    )
    batch_size = training_batch_size(
        smoke_test=cfg.smoke_test,
        default_batch_size=cfg.batch_size,
    )
    train_loader = DataLoader(
        dataset_subset(train_dataset, smoke_test=cfg.smoke_test),  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset_subset(val_dataset, smoke_test=cfg.smoke_test),  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    d_out = train_dataset.label_shape[0]
    backbone, expansion = _load_backbone(cfg.architecture.backbone_type)
    classifier_bundle, _ = load_bundle(
        Path(cfg.pretrained_model_bundle_path),
        map_location=device,
    )
    classifier_bundle = cast(
        "GreedyAFAClassifier",
        cast("object", classifier_bundle),
    )
    predictor = classifier_bundle.predictor.to(device)
    arch = classifier_bundle.architecture
    image_size = arch["image_size"]
    patch_size = arch["patch_size"]
    assert image_size % patch_size == 0
    mask_width = arch["mask_width"]
    n_patches = int(mask_width) ** 2
    n_selections = unmasker.get_n_selections(train_dataset.feature_shape)
    assert n_selections == n_patches
    selector = ConvNet(backbone, expansion).to(device)
    mask_layer = MaskLayer2d(
        mask_width=mask_width, patch_size=patch_size, append=False
    )
    x0, _ = next(iter(train_loader))
    with torch.no_grad():
        logits0 = selector(
            mask_layer(
                x0.to(device), torch.zeros(len(x0), n_patches, device=device)
            )
        )
    assert logits0.shape[1] == n_patches, (
        f"Selector outputs {logits0.shape[1]} != n_patches {n_patches}"
    )
    gdfs = GreedyDynamicSelection(
        selector=selector,
        predictor=predictor,
        mask_layer=mask_layer,
        initializer=initializer,
        unmasker=unmasker,
    ).to(device)
    assert cfg.min_lr is not None, "min_lr must be set for image training"
    gdfs.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        min_lr=cfg.min_lr,
        nepochs=cfg.nepochs,
        max_features=cfg.hard_budget,
        loss_fn=nn.CrossEntropyLoss(),
        patience=cfg.patience,
        temp_steps=1 if cfg.smoke_test else 5,
        verbose=True,
        metric_logger=metric_logger,
        metric_prefix="gdfs",
    )
    afa_method = GDFSAFAMethod(
        selector=gdfs.selector.cpu(),
        predictor=gdfs.predictor.cpu(),
        device=torch.device("cpu"),
        modality="image",
        n_patches=n_patches,
        d_out=d_out,
        backbone_type=cfg.architecture.backbone_type,
    )
    afa_method.image_size = image_size
    afa_method.patch_size = patch_size
    afa_method.mask_width = mask_width
    save_bundle(
        obj=afa_method,
        path=Path(cfg.save_path),
        metadata={"config": asdict(cfg)},
    )
    log.info(f"GDFS method saved to: {cfg.save_path}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
