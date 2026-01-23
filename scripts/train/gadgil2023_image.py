import gc
import logging
from pathlib import Path
from typing import cast

import hydra
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from afabench.afa_discriminative.afa_methods import (
    CMIEstimator,
    Gadgil2023AFAMethod,
)
from afabench.afa_discriminative.models import (
    ConvNet,
    ResNet18Backbone,
    resnet18,
)
from afabench.afa_discriminative.models import GreedyAFAClassifier
from afabench.afa_discriminative.utils import MaskLayer2d, afa_discriminative_training_prep
from afabench.common.config_classes import Gadgil2023Training2DConfig
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.utils import set_seed

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/gadgil2023",
    config_name="config",
)
def main(cfg: Gadgil2023Training2DConfig):
    log.debug(cfg)
    print(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    train_dataset, val_dataset, initializer, unmasker, _ = (
        afa_discriminative_training_prep(
            train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
            val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
            initializer_cfg=cfg.initializer,
            unmasker_cfg=cfg.unmasker,
        )
    )
    d_out = train_dataset.label_shape[0]
    train_loader = DataLoader(
        train_dataset, # pyright: ignore[reportArgumentType]
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, # pyright: ignore[reportArgumentType]
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    base = resnet18(pretrained=True)
    backbone, expansion = ResNet18Backbone(base)
    classifier_bundle, _ = load_bundle(
        Path(cfg.pretrained_model_bundle_path),
        map_location=device,
    )
    classifier_bundle = cast(
        GreedyAFAClassifier, classifier_bundle,
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

    value_network = ConvNet(backbone, expansion).to(device)

    mask_layer = MaskLayer2d(
        mask_width=mask_width, patch_size=patch_size, append=False
    )
    x0, _ = next(iter(train_loader))
    with torch.no_grad():
        logits0 = value_network(
            mask_layer(
                x0.to(device), torch.zeros(len(x0), n_patches, device=device)
            )
        )
    assert logits0.shape[1] == n_patches, (
        f"Value Network outputs {logits0.shape[1]} != n_patches {n_patches}"
    )

    greedy_cmi_estimator = CMIEstimator(
        value_network=value_network,
        predictor=predictor,
        mask_layer=mask_layer,
        initializer=initializer,
        unmasker=unmasker,
    ).to(device)
    greedy_cmi_estimator.fit(
        train_loader,
        val_loader,
        lr=cfg.lr,
        min_lr=cfg.min_lr,
        nepochs=cfg.nepochs,
        max_features=cfg.hard_budget,
        eps=cfg.eps,
        loss_fn=nn.CrossEntropyLoss(reduction="none"),
        val_loss_fn=Accuracy(task="multiclass", num_classes=d_out).to(device),
        val_loss_mode="max",
        eps_decay=cfg.eps_decay,
        eps_steps=cfg.eps_steps,
        patience=cfg.patience,
        feature_costs=None,
    )

    afa_method = Gadgil2023AFAMethod(
        value_network=greedy_cmi_estimator.value_network.cpu(),
        predictor=greedy_cmi_estimator.predictor.cpu(),
        device=torch.device("cpu"),
        modality="image",
        n_patches=n_patches,
        d_out=d_out,
    )
    afa_method.image_size = image_size
    afa_method.patch_size = patch_size
    afa_method.mask_width = mask_width

    save_bundle(
        obj=afa_method,
        path=Path(cfg.save_path),
        metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
    )

    log.info(f"Gadgil2023 method saved to: {cfg.save_path}")

    gc.collect()  # Force Python GC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
        torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
