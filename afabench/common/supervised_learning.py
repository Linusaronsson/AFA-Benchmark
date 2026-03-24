import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from afabench.afa_rl.common.dataset_utils import DataModuleFromDatasets
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.datasets.utils import flatten_features_collate
from afabench.common.torch_bundle import TorchModelBundle
from afabench.common.utils import configure_runtime_for_device

if TYPE_CHECKING:
    from torch.utils.data.dataset import Dataset

    from afabench.common.custom_types import Features, Label


from afabench.common.config_classes import (
    SupervisedLearningConfig,
)
from afabench.common.custom_types import AFADataset

log = logging.getLogger(__name__)


class ModelCheckpointWithMinBatches(ModelCheckpoint):
    def __init__(self, min_batches: int = 0, *args, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.min_batches: int = min_batches
        self._batches_seen: int = 0

    @override
    def on_train_batch_end(
        self,
        trainer,  # noqa: ANN001
        pl_module,  # noqa: ANN001
        outputs,  # noqa: ANN001
        batch,  # noqa: ANN001
        batch_idx,  # noqa: ANN001
    ) -> None:
        self._batches_seen += 1

    @override
    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        if self._batches_seen >= self.min_batches:
            super().on_validation_end(trainer, pl_module)
        # else: skip updating best model


class EarlyStoppingWithMinBatches(EarlyStopping):
    """Add min epoch functionality to EarlyStopping."""

    def __init__(self, min_batches: int = 0, *args, **kwargs):  # noqa: ANN002, ANN003
        super().__init__(*args, **kwargs)
        self.min_batches: int = min_batches
        self._batches_seen: int = 0

    @override
    def on_train_batch_end(
        self,
        trainer,  # noqa: ANN001
        pl_module,  # noqa: ANN001
        outputs,  # noqa: ANN001
        batch,  # noqa: ANN001
        batch_idx,  # noqa: ANN001
    ) -> None:
        self._batches_seen += 1

    @override
    def on_validation_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        if self._batches_seen >= self.min_batches:
            super().on_validation_end(trainer, pl_module)
        # else: do nothing, don't check for early stopping yet


def supervised_learning(
    train_dataset_bundle_path: Path,
    val_dataset_bundle_path: Path,
    save_path: Path,
    cfg: SupervisedLearningConfig,
    model_fn: Callable[[AFADataset], pl.LightningModule],
    metric_to_monitor: str,  # what we want to optimize
    monitor_mode: str,  # whether we want to minimize ("min") or maximize ("max") metric_to_monitor
    *,
    use_wandb: bool = False,
    device: str | None = None,
    metadata_to_save_in_bundle: dict[str, Any] | None = None,
) -> None:
    """
    Do supervised learning for a pytorch lightning model.

    Currently assumes that the model expects 1D (flattened) features.
    """
    if device is None:
        device = "cpu"
    device = configure_runtime_for_device(device)
    if metadata_to_save_in_bundle is None:
        metadata_to_save_in_bundle = {}
    log.info("Loading datasets...")
    train_dataset, train_dataset_manifest = load_bundle(
        Path(train_dataset_bundle_path),
    )
    train_dataset = cast("AFADataset", cast("object", train_dataset))
    _train_features, _train_labels = train_dataset.get_all_data()
    val_dataset, _val_dataset_metadata = load_bundle(
        Path(val_dataset_bundle_path),
    )
    val_dataset = cast("AFADataset", cast("object", val_dataset))
    datamodule = DataModuleFromDatasets(
        train_dataset=cast(
            "Dataset[tuple[Features, Label]]", cast("object", train_dataset)
        ),
        val_dataset=cast(
            "Dataset[tuple[Features, Label]]", cast("object", val_dataset)
        ),
        batch_size=cfg.batch_size,
        collate_fn=flatten_features_collate(
            n_feature_dims=len(train_dataset.feature_shape)
        ),
    )
    log.info("Loaded datasets.")

    log.info("Creating model...")
    lit_model = model_fn(train_dataset)
    lit_model = lit_model.to(device)
    log.info("Created model.")

    log.info("Starting training...")
    checkpoint_callback = ModelCheckpointWithMinBatches(
        min_batches=cfg.checkpoint_earliest_batch,
        monitor=metric_to_monitor,
        save_top_k=1,
        mode=monitor_mode,
    )
    early_stopping_callback = EarlyStoppingWithMinBatches(
        min_batches=cfg.early_stopping_min_batches,
        monitor=metric_to_monitor,
        min_delta=cfg.early_stopping_min_delta,
        patience=cfg.early_stopping_patience,
        mode=monitor_mode,
        verbose=True,
    )
    logger = WandbLogger(save_dir="extra/logs/wandb") if use_wandb else False
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        logger=logger,
        accelerator=device,
        devices=1,
        callbacks=[checkpoint_callback, early_stopping_callback],
        # Run validation every `cfg.val_check_interval` training batches if set.
        # If None, Lightning will validate at the end of each epoch.
        val_check_interval=cfg.val_check_interval,
        check_val_every_n_epoch=None,
        default_root_dir="extra/logs/lightning",
    )

    try:
        trainer.fit(lit_model, datamodule)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Finished training.")

        assert trainer.checkpoint_callback is not None
        best_model_path: str | None = (
            trainer.checkpoint_callback.best_model_path  # pyright: ignore[reportAttributeAccessIssue]
        )
        if (
            best_model_path is not None
            and len(best_model_path) != 0
            and Path(best_model_path).exists()
        ):
            log.info("Resetting state to best model...")
            # Reset to best model found during training
            lit_model.load_state_dict(
                torch.load(
                    trainer.checkpoint_callback.best_model_path,  # pyright: ignore[reportAttributeAccessIssue]
                    map_location="cpu",
                )["state_dict"]
            )
        else:
            log.warning("No best model found. Keeping current model...")
        log.info("Finished setting model state.")

        log.info("Saving model...")

        # Create general model bundle wrapper
        model_bundle = TorchModelBundle(lit_model)

        # Save using bundle format
        bundle_path = Path(save_path)
        if bundle_path.suffix != ".bundle":
            bundle_path = bundle_path.with_suffix(".bundle")
        metadata = {
            "dataset_class_name": train_dataset_manifest["class_name"],
        } | metadata_to_save_in_bundle
        save_bundle(model_bundle, bundle_path, metadata)
        log.info(f"Saved best model to {bundle_path}")
