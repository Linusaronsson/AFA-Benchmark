import logging
from collections.abc import Callable
from dataclasses import replace
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
from afabench.core.utils import get_class_frequencies
from afabench.training.inputs import load_inputs
from afabench.training.run import save_result, training_run
from afabench.training.smoke_test import limit_supervised_learning
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
    torch.set_float32_matmul_precision("medium")
    cfg = replace(
        cfg,
        supervised_learning=limit_supervised_learning(
            cfg.supervised_learning, smoke_test=cfg.smoke_test
        ),
    )

    with training_run(cfg, "pretraining", tags=["jafa"], config=cfg):
        inputs = load_inputs(cfg)
        model_bundle = supervised_learning(
            train_dataset=inputs.train_dataset(),
            val_dataset=inputs.val_dataset(),
            cfg=cfg.supervised_learning,
            model_fn=get_jafa_model_fn(cfg=cfg),
            metric_to_monitor="val_loss_many_observations",
            monitor_mode="min",
            use_wandb=cfg.use_wandb,
            device=cfg.device,
        )
        save_result(model_bundle, cfg, cfg, stage="pretraining")


if __name__ == "__main__":
    main()
