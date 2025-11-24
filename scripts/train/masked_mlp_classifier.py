import torch
import wandb
import hydra
import logging
import lightning as pl

from pathlib import Path
from omegaconf import OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from afabench.eval.utils import plot_metrics
from afabench.eval.hard_budget import eval_afa_method
from afabench.common.models import LitMaskedMLPClassifier
from afabench.afa_rl.datasets import DataModuleFromDatasets
from afabench.common.classifiers import WrappedMaskedMLPClassifier
from afabench.common.afa_methods import RandomClassificationAFAMethod
from afabench.common.config_classes import TrainMaskedMLPClassifierConfig

from afabench.common.utils import (
    get_class_probabilities,
    load_dataset,
    set_seed,
)

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

    run = wandb.init(
        group="train_masked_mlp_classifier",
        job_type="train_classifier",
        config=OmegaConf.to_container(cfg, resolve=True),
        dir="extra/wandb",
    )

    log.info(f"W&B run initialized: {run.name} ({run.id})")
    log.info(f"W&B run URL: {run.url}")

    # Load dataset
    train_dataset, val_dataset, _, dataset_metadata = load_dataset(
        cfg.dataset_artifact_name
    )
    datamodule = DataModuleFromDatasets(
        train_dataset, val_dataset, batch_size=cfg.batch_size
    )

    n_features = train_dataset.features.shape[-1]
    n_classes = train_dataset.labels.shape[-1]

    train_class_probabilities = get_class_probabilities(train_dataset.labels)
    log.debug(f"Class probabilities: {train_class_probabilities}")

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

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss_many_observations",
        save_top_k=1,
        mode="min",
    )

    logger = WandbLogger(save_dir="extra/wandb")
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        logger=logger,
        accelerator=cfg.device,
        devices=1,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
    )

    try:
        trainer.fit(lit_model, datamodule, ckpt_path=None)
    except KeyboardInterrupt:
        pass
    finally:
        # Load best checkpoint
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        best_lit_model = LitMaskedMLPClassifier.load_from_checkpoint(
            best_checkpoint,
            n_features=n_features,
            n_classes=n_classes,
            num_cells=cfg.num_cells,
            dropout=cfg.dropout,
            class_probabilities=train_class_probabilities,
            max_masking_probability=cfg.max_masking_probability,
            lr=cfg.lr,
        )
        best_model = best_lit_model.classifier
        wrapped_classifier = WrappedMaskedMLPClassifier(
            module=best_model, device=torch.device(cfg.device)
        )

        if cfg.evaluate_final_performance:
            afa_method = RandomClassificationAFAMethod(
                afa_classifier=wrapped_classifier, device=torch.device("cpu")
            )
            metrics = eval_afa_method(
                afa_select_fn=afa_method.select,
                dataset=val_dataset,
                budget=n_features,
                afa_predict_fn=afa_method.predict,
                only_n_samples=cfg.eval_only_n_samples,
            )
            fig = plot_metrics(metrics)
            run.log({"metrics_plot": fig})

        # Save to local filesystem
        dataset_name = cfg.dataset_artifact_name.split("/")[0]
        dataset_split = cfg.dataset_artifact_name.split("_")[-1]
        classifier_name = (
            f"masked_mlp_classifier-{dataset_name}_split_{dataset_split}"
        )
        classifier_dir = Path("extra/classifiers") / classifier_name
        classifier_dir.mkdir(parents=True, exist_ok=True)

        # Save classifier
        wrapped_classifier.save(classifier_dir / "classifier.pt")

        # Save metadata
        import json

        metadata = {
            "classifier_class_name": wrapped_classifier.__class__.__name__,
            "dataset_artifact_name": cfg.dataset_artifact_name,
            "dataset_type": dataset_metadata["dataset_type"],
            "seed": cfg.seed,
            "num_cells": list(cfg.num_cells),
            "dropout": cfg.dropout,
        }
        with open(classifier_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        log.info(f"Classifier saved to: {classifier_dir}")
        run.finish()


if __name__ == "__main__":
    main()
