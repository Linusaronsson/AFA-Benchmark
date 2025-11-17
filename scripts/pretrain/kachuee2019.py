import gc
import wandb
import hydra
import torch
import logging
import lightning as pl

from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from afabench.afa_rl.datasets import DataModuleFromDatasets
from afabench.common.config_classes import Kachuee2019PretrainConfig
from afabench.afa_rl.kachuee2019.utils import get_kachuee2019_model_from_config

from afabench.common.utils import (
    get_class_probabilities,
    load_dataset_artifact,
    set_seed,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/pretrain/kachuee2019",
    config_name="config",
)
def main(cfg: Kachuee2019PretrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    run = wandb.init(
        group="pretrain_kachuee2019",
        job_type="pretraining",
        config=OmegaConf.to_container(cfg, resolve=True),  # pyright: ignore
        tags=["kachuee2019"],
        dir="extra/wandb",
    )

    # Log W&B run URL
    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Load dataset artifact
    train_dataset, val_dataset, _, _ = load_dataset_artifact(
        cfg.dataset_artifact_name
    )
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=cfg.batch_size
    )

    # Get the number of features and classes from the dataset
    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(
        f"Class probabilities in training set: {train_class_probabilities}"
    )
    lit_model = get_kachuee2019_model_from_config(
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
        # Save best model as wandb artifact
        best_checkpoint = trainer.checkpoint_callback.best_model_path  # pyright: ignore
        pretrained_model_artifact = wandb.Artifact(
            name=f"pretrain_kachuee2019-{
                cfg.dataset_artifact_name.split(':')[0]
            }",
            type="pretrained_model",
        )
        pretrained_model_artifact.add_file(
            local_path=best_checkpoint, name="model.pt"
        )
        run.log_artifact(
            pretrained_model_artifact, aliases=cfg.output_artifact_aliases
        )
        run.finish()

        gc.collect()  # Force Python GC
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Release cached memory held by PyTorch CUDA allocator
            torch.cuda.synchronize()  # Optional, wait for CUDA ops to finish


if __name__ == "__main__":
    main()
