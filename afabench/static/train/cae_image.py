import gc
import logging
from pathlib import Path
from typing import cast

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from afabench.afa_discriminative.models import (
    Predictor,
    ResNet18Backbone,
    resnet18,
)
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import CAETraining2DConfig
from afabench.common.custom_types import AFADataset  # noqa: TC001
from afabench.common.utils import set_seed
from afabench.static.models import BaseModel
from afabench.static.static_methods import (
    ConcreteMask2d,
    DifferentiableSelector,
    StaticBaseMethod,
)
from afabench.static.utils import make_masked_collate

log = logging.getLogger(__name__)


def train_image(cfg: CAETraining2DConfig) -> None:  # noqa: PLR0915
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    torch.set_float32_matmul_precision("medium")
    if cfg.smoke_test:
        cfg.selector.nepochs = 1
        cfg.selector.patience = 1
        cfg.classifier.nepochs = 1

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
    image_size = cfg.image_size
    patch_size = cfg.patch_size
    assert image_size % patch_size == 0, (
        "image_size must be divisible by patch_size"
    )
    mask_width = image_size // patch_size

    base = resnet18(pretrained=True)
    backbone, expansion = ResNet18Backbone(base)
    model = Predictor(backbone, expansion, num_classes=d_out).to(device)
    selector_layer = ConcreteMask2d(
        width=mask_width,
        patch_size=patch_size,
        num_select=cfg.hard_budget,
    )
    diff_selector = DifferentiableSelector(
        model=model,
        selector_layer=selector_layer,
    ).to(device)
    diff_selector.fit(
        train_loader,
        val_loader,
        lr=cfg.selector.lr,
        nepochs=cfg.selector.nepochs,
        loss_fn=nn.CrossEntropyLoss(),
        patience=cfg.selector.patience,
        verbose=True,
    )

    logits = selector_layer.logits.cpu().data.numpy()
    ranked_patches = np.sort(logits.argmax(axis=1))

    if len(np.unique(ranked_patches)) != cfg.hard_budget:
        print(
            f"{len(np.unique(ranked_patches))} selected instead of {
                cfg.hard_budget
            }, appending extras"
        )
    num_extras = cfg.hard_budget - len(np.unique(ranked_patches))
    remaining_patches = np.setdiff1d(np.arange(mask_width**2), ranked_patches)
    ranked_patches = np.sort(
        np.concatenate(
            [np.unique(ranked_patches), remaining_patches[:num_extras]]
        )
    )

    predictors: dict[int, nn.Module] = {}
    selected_history: dict[int, list[int]] = {}

    num_features = list(range(1, cfg.hard_budget + 1))
    for num in num_features:
        selected_patches = ranked_patches[:num]
        selected_history[num] = selected_patches.tolist()
        patch_mask = torch.zeros(
            mask_width**2, dtype=torch.float32, device="cpu"
        )
        idx = torch.as_tensor(selected_patches, dtype=torch.long, device="cpu")
        patch_mask[idx] = 1.0
        # Assume [B, C, H, W] input
        patch_mask = (
            patch_mask.view(mask_width, mask_width).unsqueeze(0).unsqueeze(0)
        )
        if patch_size > 1:
            patch_mask = torch.nn.Upsample(
                scale_factor=patch_size,
                mode="nearest",
            )(patch_mask)

        masked_train_loader = DataLoader(
            train_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=make_masked_collate(patch_mask),
        )
        masked_val_loader = DataLoader(
            val_dataset,  # pyright: ignore[reportArgumentType]
            batch_size=cfg.batch_size,
            pin_memory=True,
            collate_fn=make_masked_collate(patch_mask),
        )

        base = resnet18(pretrained=True)
        backbone, expansion = ResNet18Backbone(base)
        model = Predictor(backbone, expansion, num_classes=d_out).to(device)
        predictor = BaseModel(model).to(device)
        predictor.fit(
            masked_train_loader,
            masked_val_loader,
            lr=cfg.classifier.lr,
            nepochs=cfg.classifier.nepochs,
            loss_fn=nn.CrossEntropyLoss(),
            verbose=True,
        )

        predictors[num] = model

    static_method = StaticBaseMethod(
        selected_history=selected_history,
        predictors=predictors,
        image_size=image_size,
        patch_size=patch_size,
        device=device,
    )

    save_bundle(
        obj=static_method,
        path=Path(cfg.save_path),
        metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
    )
    log.info(f"CAE method saved to: {cfg.save_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
