import hydra
import torch
import logging
import lightning as pl

from pathlib import Path
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from afabench.common.models import LitMaskedMLPClassifier
from afabench.afa_rl.common.dataset_utils import DataModuleFromDatasets
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.utils import get_class_frequencies, set_seed
from afabench.common.classifiers import WrappedMaskedMLPClassifier
from afabench.common.config_classes import TrainMaskedMLPClassifierConfig

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/classifiers/masked_mlp_classifier",
    config_name="config",
)
def main(cfg: TrainMaskedMLPClassifierConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device(cfg.device)

    # Handle smoke test
    if cfg.smoke_test:
        log.info("Smoke test mode: reducing epochs and batch size")
        cfg.epochs = 2
        cfg.batch_size = min(cfg.batch_size, 32)

    # Load datasets via bundle system
    train_dataset, train_manifest = load_bundle(Path(cfg.train_dataset_path))
    val_dataset, _ = load_bundle(Path(cfg.val_dataset_path))

    dataset_name = train_manifest["class_name"].replace("Dataset", "").lower()
    log.info(f"Dataset: {train_manifest['class_name']}")
    log.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=cfg.batch_size
    )

    # Get dimensions (flatten for MLP)
    feature_shape = train_dataset.feature_shape
    n_features = feature_shape.numel()
    n_classes = train_dataset.label_shape[0]

    # Class weights
    _, train_labels = train_dataset.get_all_data()
    train_class_probabilities = get_class_frequencies(train_labels)

    # Model
    lit_model = LitMaskedMLPClassifier(
        n_features=n_features,
        n_classes=n_classes,
        num_cells=tuple(cfg.num_cells),
        dropout=cfg.dropout,
        class_probabilities=train_class_probabilities,
        min_masking_probability=cfg.min_masking_probability,
        max_masking_probability=cfg.max_masking_probability,
        lr=cfg.lr,
    )

    # Training
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_many_observations",
        save_top_k=1,
        mode="min",
    )

    log_dir = Path(cfg.save_path).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=CSVLogger(save_dir=str(log_dir), name="masked_mlp_classifier"),
        accelerator=cfg.device,
        devices=1,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    trainer.fit(lit_model, datamodule)

    # Load best checkpoint
    best_checkpoint = checkpoint_callback.best_model_path
    if not best_checkpoint:
        raise RuntimeError("No checkpoint saved during training")

    best_lit_model = LitMaskedMLPClassifier.load_from_checkpoint(
        best_checkpoint,
        n_features=n_features,
        n_classes=n_classes,
        num_cells=tuple(cfg.num_cells),
        dropout=cfg.dropout,
        class_probabilities=train_class_probabilities,
        min_masking_probability=cfg.min_masking_probability,
        max_masking_probability=cfg.max_masking_probability,
        lr=cfg.lr,
    )

    # Wrap and save
    wrapped_classifier = WrappedMaskedMLPClassifier(
        module=best_lit_model.classifier, device=device
    )

    save_bundle(
        obj=wrapped_classifier,
        path=Path(cfg.save_path),
        metadata={
            "train_dataset_path": cfg.train_dataset_path,
            "val_dataset_path": cfg.val_dataset_path,
            "dataset_name": dataset_name,
            "seed": cfg.seed,
            "num_cells": list(cfg.num_cells),
            "dropout": cfg.dropout,
            "n_features": n_features,
            "n_classes": n_classes,
            "epochs": cfg.epochs,
            "min_masking_probability": cfg.min_masking_probability,
            "max_masking_probability": cfg.max_masking_probability,
        },
    )
    log.info(f"Saved classifier to: {cfg.save_path}")


if __name__ == "__main__":
    main()
