import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import cast

import hydra
import lightning as pl
import torch
from omegaconf import OmegaConf

from afabench.components.methods.rl.jafa.config import JAFAPretrainConfig
from afabench.components.methods.rl.jafa.models import (
    JAFAEmbedder,
    JAFAMLPClassifier,
    LitJAFAEmbedderClassifier,
    ReadProcessEncoder,
)
from afabench.core.types import AFADataset
from afabench.core.utils import (
    get_class_frequencies,
    initialize_wandb_run,
    set_seed,
)
from afabench.training.supervised_learning import supervised_learning

log = logging.getLogger(__name__)


def get_jafa_model_fn(
    cfg: JAFAPretrainConfig,
) -> Callable[[AFADataset], pl.LightningModule]:
    def f(dataset: AFADataset) -> pl.LightningModule:
        _features, labels = dataset.get_all_data()
        class_probabilities = get_class_frequencies(labels)
        n_features = dataset.feature_shape.numel()
        n_classes = dataset.label_shape.numel()
        encoder = ReadProcessEncoder(
            set_element_size=n_features
            + 1,  # state contains one value and one index
            output_size=cfg.encoder.output_size,
            reading_block_cells=tuple(cfg.encoder.reading_block_cells),
            writing_block_cells=tuple(cfg.encoder.writing_block_cells),
            memory_size=cfg.encoder.memory_size,
            processing_steps=cfg.encoder.processing_steps,
            dropout=cfg.encoder.dropout,
        )
        embedder = JAFAEmbedder(encoder)
        classifier = JAFAMLPClassifier(
            cfg.encoder.output_size, n_classes, tuple(cfg.classifier.num_cells)
        )
        lit_model = LitJAFAEmbedderClassifier(
            embedder=embedder,
            classifier=classifier,
            class_probabilities=class_probabilities,
            min_masking_probability=cfg.min_masking_probability,
            max_masking_probability=cfg.max_masking_probability,
            lr=cfg.lr,
        )
        return lit_model

    return f


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain_model/jafa",
    config_name="config",
)
def main(cfg: JAFAPretrainConfig) -> None:
    cfg = cast("JAFAPretrainConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        _run = initialize_wandb_run(
            cfg=asdict(cfg),
            job_type="pretraining",
            tags=["jafa"],
        )

    # If smoke test, override some options
    if cfg.smoke_test:
        log.info("Smoke test detected.")
        cfg.supervised_learning.max_epochs = 1
        cfg.supervised_learning.limit_train_batches = 2
        cfg.supervised_learning.limit_val_batches = 2

    supervised_learning(
        train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
        val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
        save_path=Path(cfg.save_path),
        cfg=cfg.supervised_learning,
        model_fn=get_jafa_model_fn(cfg=cfg),
        metric_to_monitor="val_loss_many_observations",
        monitor_mode="min",
        use_wandb=cfg.use_wandb,
        device=cfg.device,
        metadata_to_save_in_bundle={
            "train_dataset_bundle_path": cfg.train_dataset_bundle_path,
            "seed": cfg.seed,
            "config": asdict(cfg),
        },
    )


if __name__ == "__main__":
    main()
