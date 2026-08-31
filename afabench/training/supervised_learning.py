import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, override

import lightning as pl
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from afabench.components.methods.rl.common.dataset_utils import (
    DataModuleFromDatasets,
)
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.core.types import AFADataset
from afabench.training.config import SupervisedLearningConfig
from afabench.training.tensor_batches import (
    TensorBatchDataset,
    passthrough_batch,
)

if TYPE_CHECKING:
    from torch.utils.data.dataset import Dataset

    from afabench.core.types import Features, Label


log = logging.getLogger(__name__)


def dataset_source_availability(dataset: AFADataset) -> torch.Tensor:
    availability = getattr(dataset, "source_availability", None)
    if availability is None:
        msg = "Dataset does not define source availability."
        raise ValueError(msg)
    return cast("torch.Tensor", availability)


def lightning_root() -> Path:
    """Keep disposable Lightning files on node-local storage when available."""
    return Path(os.environ.get("SNIC_TMP", "extra/logs/lightning"))


def ensure_finite_module_state(module: torch.nn.Module) -> None:
    nonfinite_state = [
        name
        for name, tensor in module.state_dict().items()
        if (tensor.is_floating_point() or tensor.is_complex())
        and not torch.isfinite(tensor).all()
    ]
    if nonfinite_state:
        msg = f"Refusing to save non-finite model state: {nonfinite_state}"
        raise FloatingPointError(msg)


def _tensor_dataset(
    dataset: AFADataset,
    row_extras_fn: Callable[[AFADataset], torch.Tensor] | None,
) -> TensorBatchDataset:
    features, labels = dataset.get_all_data()
    tensors = [features.flatten(start_dim=1), labels]
    if row_extras_fn is not None:
        row_extra = row_extras_fn(dataset)
        if len(dataset) != len(row_extra):
            msg = "Row extras must have one row per dataset instance."
            raise ValueError(msg)
        tensors.append(row_extra.flatten(start_dim=1))
    return TensorBatchDataset(*tensors)


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
    row_extras_fn: Callable[[AFADataset], torch.Tensor] | None = None,
) -> None:
    """
    Do supervised learning for a pytorch lightning model.

    Currently assumes that the model expects 1D (flattened) features.
    """
    if device is None:
        device = "cpu"
    if metadata_to_save_in_bundle is None:
        metadata_to_save_in_bundle = {}
    log.info("Loading datasets...")
    train_dataset, train_dataset_manifest = load_bundle(
        Path(train_dataset_bundle_path),
    )
    train_dataset = cast("AFADataset", cast("object", train_dataset))
    val_dataset, _val_dataset_metadata = load_bundle(
        Path(val_dataset_bundle_path),
    )
    val_dataset = cast("AFADataset", cast("object", val_dataset))

    datamodule = DataModuleFromDatasets(
        train_dataset=cast(
            "Dataset[tuple[Features, Label]]",
            cast("object", _tensor_dataset(train_dataset, row_extras_fn)),
        ),
        val_dataset=cast(
            "Dataset[tuple[Features, Label]]",
            cast("object", _tensor_dataset(val_dataset, row_extras_fn)),
        ),
        batch_size=cfg.batch_size,
        collate_fn=passthrough_batch,
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
        default_root_dir=lightning_root(),
        enable_progress_bar=sys.stderr.isatty(),
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

        ensure_finite_module_state(lit_model)

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
