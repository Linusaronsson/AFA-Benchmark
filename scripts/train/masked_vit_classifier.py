import gc
import logging
from pathlib import Path

import hydra
import timm
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from afabench.afa_discriminative.utils import MaskLayer2d, afa_discriminative_training_prep
from afabench.common.classifiers import WrappedMaskedViTClassifier
from afabench.common.config_classes import TrainMaskedViTClassifierConfig
from afabench.common.models import MaskedViTClassifier, MaskedViTTrainer
from afabench.common.utils import set_seed
from afabench.common.bundle import save_bundle

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/classifiers/masked_vit_classifier",
    config_name="config",
)
def main(cfg: TrainMaskedViTClassifierConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)
    if cfg.smoke_test:
        cfg.epochs = 1
        cfg.patience = 1

    train_dataset, val_dataset, _, _, _ = (
        afa_discriminative_training_prep(
            train_dataset_bundle_path=Path(cfg.train_dataset_path),
            val_dataset_bundle_path=Path(cfg.val_dataset_path),
            initializer_cfg=cfg.initializer,
            unmasker_cfg=cfg.unmasker,
            seed=cfg.seed,
        )
    )
    d_out = train_dataset.label_shape[0]
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
    backbone = timm.create_model(cfg.model_name, pretrained=True)
    model = MaskedViTClassifier(backbone=backbone, num_classes=d_out)

    assert cfg.image_size % cfg.patch_size == 0
    mask_width = cfg.image_size // cfg.patch_size
    mask_layer = MaskLayer2d(
        mask_width=mask_width, patch_size=cfg.patch_size, append=False
    )
    trainer = MaskedViTTrainer(model, mask_layer).to(device)

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        lr=cfg.lr,
        nepochs=cfg.epochs,
        loss_fn=nn.CrossEntropyLoss(),
        val_loss_fn=nn.CrossEntropyLoss(),
        val_loss_mode="min",
        patience=cfg.patience,
        min_lr=cfg.min_lr,
        min_mask=cfg.min_masking_probability,
        max_mask=cfg.max_masking_probability,
        logger=log,
    )

    wrapped_classifier = WrappedMaskedViTClassifier(
        module=model,
        device=device,
        pretrained_model_name=cfg.model_name,
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
    )
    save_bundle(
        obj=wrapped_classifier,
        path=Path(cfg.save_path),
        metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
    )

    log.info(f"Masked ViT classifier saved to: {cfg.save_path}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
