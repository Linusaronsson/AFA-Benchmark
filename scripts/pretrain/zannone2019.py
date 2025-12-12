import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf

from afabench.afa_rl.datasets import DataModuleFromDatasets
from afabench.afa_rl.zannone2019.utils import get_zannone2019_model_from_config
from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import Zannone2019PretrainConfig
from afabench.common.torch_bundle import TorchModelBundle
from afabench.common.utils import (
    get_class_frequencies,
    initialize_wandb_run,
    set_seed,
)

if TYPE_CHECKING:
    from torch.utils.data.dataset import Dataset

    from afabench.common.custom_types import AFADataset, Features, Label

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain/zannone2019",
    config_name="config",
)
def main(cfg: Zannone2019PretrainConfig) -> None:  # noqa: PLR0915
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        run = initialize_wandb_run(
            cfg=cfg, job_type="pretraining", tags=["zannone2019"]
        )
    else:
        run = None

    # If smoke test, override some options
    if cfg.smoke_test:
        log.info("Smoke test detected.")
        cfg.epochs = 1
        cfg.limit_train_batches = 2
        cfg.limit_val_batches = 2

    log.info("Loading datasets...")
    train_dataset, train_dataset_manifest = load_bundle(
        Path(cfg.train_dataset_bundle_path),
    )
    train_dataset = cast("AFADataset", cast("object", train_dataset))
    train_features, train_labels = train_dataset.get_all_data()
    val_dataset, val_dataset_metadata = load_bundle(
        Path(cfg.val_dataset_bundle_path),
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
    )
    log.info("Loaded datasets.")

    log.info("Creating model...")
    train_class_probabilities = get_class_frequencies(train_labels)
    assert len(train_dataset.label_shape) == 1, "Only 1D label supported"
    lit_model = get_zannone2019_model_from_config(
        cfg,
        feature_shape=train_dataset.feature_shape,
        n_classes=train_dataset.label_shape.numel(),
        class_probabilities=train_class_probabilities,
    )
    lit_model = lit_model.to(cfg.device)
    log.info("Created model.")

    log.info("Starting training...")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_many_observations",
        save_top_k=1,
        mode="min",
    )
    logger = WandbLogger(save_dir="extra/wandb") if cfg.use_wandb else False
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        accelerator=cfg.device,
        devices=1,
        callbacks=[checkpoint_callback],
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )

    try:
        trainer.fit(lit_model, datamodule)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("Finished training.")

        log.info("Saving best model...")
        best_checkpoint_path = Path(
            trainer.checkpoint_callback.best_model_path  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess]
        )

        # Load the best model checkpoint
        best_lit_model = type(lit_model).load_from_checkpoint(
            best_checkpoint_path,
            # Add any required arguments for loading here if needed
            map_location="cpu",
        )

        # Create general model bundle wrapper
        model_bundle = TorchModelBundle(best_lit_model)

        # Save using bundle format
        bundle_path = Path(cfg.save_path)
        if bundle_path.suffix != ".bundle":
            bundle_path = bundle_path.with_suffix(".bundle")
        metadata = {
            "dataset_class_name": train_dataset_manifest["class_name"],
            "train_dataset_bundle_path": cfg.train_dataset_bundle_path,
            "seed": cfg.seed,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        save_bundle(model_bundle, bundle_path, metadata)
        log.info(f"Saved best model to {bundle_path}")

        if run is not None:
            run.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
