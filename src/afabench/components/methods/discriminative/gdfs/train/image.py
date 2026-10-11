import logging
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from afabench.components.methods.discriminative.common.checkpoint import (
    NetworkPairSpec,
)
from afabench.components.methods.discriminative.common.models import (
    ConvNet,
    GreedyAFAClassifier,
    ResNet18Backbone,
    resnet18,
    resnet50,
)
from afabench.components.methods.discriminative.common.trainer import (
    NetworkPairTrainer,
    TrainerSettings,
)
from afabench.components.methods.discriminative.common.utils import (
    MaskLayer2d,
)
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
    GDFSTrainingObjective,
)
from afabench.components.methods.discriminative.gdfs.config import (
    GDFSImageArchitectureConfig,
    GDFSTrainingConfig,
)
from afabench.fit.inputs import FitInputs
from afabench.fit.smoke_test import dataset_subset, training_batch_size

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
    *,
    inputs: FitInputs,
) -> GDFSAFAMethod:
    log.debug(cfg)
    assert isinstance(cfg.architecture, GDFSImageArchitectureConfig)
    assert cfg.hard_budget is not None, "hard_budget must be configured"
    assert cfg.pretrained_model_bundle_path is not None, (
        "pretrained_model_bundle_path must be configured"
    )
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)
    train_dataset = inputs.train_dataset()
    val_dataset = inputs.val_dataset()
    initializer = inputs.initializer()
    unmasker = inputs.unmasker()
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
    classifier_bundle = inputs.pretrained_model(GreedyAFAClassifier)
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
    trainer = NetworkPairTrainer(
        policy_network=selector,
        predictor=predictor,
        mask_layer=mask_layer,
        initializer=initializer,
        unmasker=unmasker,
        objective=GDFSTrainingObjective(temp_steps=1 if cfg.smoke_test else 5),
    )
    assert cfg.min_lr is not None, "min_lr must be set for image training"
    trainer.fit(
        train_loader,
        val_loader,
        TrainerSettings(
            lr=cfg.lr,
            nepochs=cfg.nepochs,
            selections_per_episode=cfg.hard_budget,
            loss_fn=nn.CrossEntropyLoss(),
            min_lr=cfg.min_lr,
            patience=cfg.patience,
        ),
        metric_logger=metric_logger,
    )
    spec = NetworkPairSpec(
        modality="image",
        d_out=d_out,
        backbone=cfg.architecture.backbone_type,
        image_size=int(image_size),
        patch_size=int(patch_size),
        mask_width=int(mask_width),
    )
    return GDFSAFAMethod(
        selector.cpu(),
        predictor.cpu(),
        spec,
        torch.device("cpu"),
    )
