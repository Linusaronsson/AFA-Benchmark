import logging
from collections.abc import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from afabench.components.methods.discriminative.common.models import (
    Predictor,
    ResNet18Backbone,
    resnet18,
    resnet50,
)
from afabench.components.methods.static.cae.config import (
    CAEImageArchitectureConfig,
    CAETrainingConfig,
)
from afabench.components.methods.static.common.models import BaseModel
from afabench.components.methods.static.common.static_methods import (
    ConcreteMask2d,
    DifferentiableSelector,
    StaticBaseMethod,
)
from afabench.components.methods.static.common.utils import (
    make_masked_collate,
)
from afabench.fit.inputs import FitInputs

log = logging.getLogger(__name__)


def train_image(  # noqa: PLR0915
    cfg: CAETrainingConfig,
    metric_logger: Callable[[dict[str, float]], None] | None = None,
    *,
    inputs: FitInputs,
) -> StaticBaseMethod:
    log.debug(cfg)
    assert isinstance(cfg.architecture, CAEImageArchitectureConfig)
    assert cfg.hard_budget is not None, "hard_budget must be configured"
    print(str(cfg))
    device = torch.device(cfg.device)
    torch.set_float32_matmul_precision("medium")
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
    image_size = cfg.architecture.image_size
    patch_size = cfg.architecture.patch_size
    assert image_size % patch_size == 0, (
        "image_size must be divisible by patch_size"
    )
    mask_width = image_size // patch_size

    backbone_type = cfg.architecture.backbone_type
    if backbone_type == "resnet18":
        base = resnet18(pretrained=True)
    elif backbone_type == "resnet50":
        base = resnet50(pretrained=True)
    else:
        msg = f"Unsupported backbone type: {backbone_type}"
        raise ValueError(msg)
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
        lr=cfg.architecture.selector.lr,
        nepochs=cfg.architecture.selector.nepochs,
        loss_fn=nn.CrossEntropyLoss(),
        patience=cfg.architecture.selector.patience,
        verbose=True,
        metric_logger=metric_logger,
        metric_prefix="cae_selector",
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

        if backbone_type == "resnet18":
            base = resnet18(pretrained=True)
        else:
            base = resnet50(pretrained=True)
        backbone, expansion = ResNet18Backbone(base)
        model = Predictor(backbone, expansion, num_classes=d_out).to(device)
        predictor = BaseModel(model).to(device)
        predictor.fit(
            masked_train_loader,
            masked_val_loader,
            lr=cfg.architecture.classifier.lr,
            nepochs=cfg.architecture.classifier.nepochs,
            loss_fn=nn.CrossEntropyLoss(),
            verbose=True,
            metric_logger=metric_logger,
            metric_prefix=f"cae_classifier/{num}_features",
        )

        predictors[num] = model

    static_method = StaticBaseMethod(
        selected_history=selected_history,
        predictors=predictors,
        image_size=image_size,
        patch_size=patch_size,
        device=device,
    )

    return static_method
