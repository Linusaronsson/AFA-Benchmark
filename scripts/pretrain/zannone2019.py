import gc
from pathlib import Path
from tempfile import TemporaryDirectory
import wandb
import hydra
import torch
import logging
import lightning as pl

from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from afabench import SAVE_PATH
from afabench.afa_rl.datasets import DataModuleFromDatasets
from afabench.common.config_classes import Zannone2019PretrainConfig
from afabench.afa_rl.zannone2019.utils import get_zannone2019_model_from_config

from afabench.common.utils import (
    get_class_probabilities,
    load_dataset,
    set_seed,
    save_artifact,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/pretrain/zannone2019",
    config_name="config",
)
def main(cfg: Zannone2019PretrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        group="pretrain_zannone2019",
        job_type="pretraining",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
        tags=["zannone2019"],
        dir="extra/wandb",
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Load dataset from filesystem
    train_dataset, val_dataset, _, dataset_metadata = load_dataset(
        cfg.dataset_artifact_name
    )

    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=cfg.batch_size
    )

    # Get the number of features and classes from the dataset
    dataset_type = dataset_metadata["dataset_type"]
    split = dataset_metadata["split_idx"]
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    log.info(f"Dataset: {dataset_type}, Split: {split}")
    log.info(f"Features: {n_features}, Classes: {n_classes}")
    log.info(f"Training samples: {len(train_dataset)}")

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(
        f"Class probabilities in training set: {train_class_probabilities}"
    )
    lit_model = get_zannone2019_model_from_config(
        cfg, n_features, n_classes, train_class_probabilities
    )
    lit_model = lit_model.to(cfg.device)

    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        # val_loss_few_observations could also work but is probably not as robust
        monitor="val_loss_many_observations",
        save_top_k=1,
        mode="min",
    )

    logger = WandbLogger(save_dir="extra/wandb")
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        accelerator=cfg.device,
        devices=1,  # Use only 1 GPU
        callbacks=[checkpoint_callback],
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
    )

    try:
        trainer.fit(lit_model, datamodule)
    except KeyboardInterrupt:
        pass
    finally:
        best_checkpoint_path = Path(
            trainer.checkpoint_callback.best_model_path
        )
        log.info(f"Best checkpoint: {best_checkpoint_path}")

        # Save as a local artifact in extra/result/shim2018/pretrain/...
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            ckpt = torch.load(best_checkpoint_path, map_location="cpu")
            torch.save(ckpt, tmp_path / "model.pt")

            artifact_identifier = (
                f"{dataset_type.lower()}_split_{split}_seed_{cfg.seed}"
            )
            artifact_dir = SAVE_PATH / artifact_identifier

            metadata = {
                "model_type": "Zannone2019",
                "dataset_type": dataset_type,
                "dataset_artifact_name": cfg.dataset_artifact_name,
                "seed": cfg.seed,
                "split_idx": split,
                "pretrain_config": OmegaConf.to_container(cfg, resolve=True),
            }

            save_artifact(
                artifact_dir=artifact_dir,
                files={"model.pt": tmp_path / "model.pt"},
                metadata=metadata,
            )

            log.info(f"Zannone2019 pretrained model saved to: {artifact_dir}")

        run.finish()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


if __name__ == "__main__":
    main()
