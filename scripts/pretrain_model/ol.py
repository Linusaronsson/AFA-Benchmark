import logging
from collections.abc import Callable
from dataclasses import replace
from typing import cast

import hydra
import lightning as pl
import torch
from omegaconf import OmegaConf

from afabench.components.methods.rl.ol.config import OLPretrainConfig
from afabench.components.methods.rl.ol.models import (
    LitOLPQModule,
    OLPQModule,
)
from afabench.core.types import AFADataset
from afabench.core.utils import get_class_frequencies
from afabench.training.inputs import TrainingInputs, load_inputs
from afabench.training.run import save_result, training_run
from afabench.training.smoke_test import limit_supervised_learning
from afabench.training.supervised_learning import supervised_learning

log = logging.getLogger(__name__)


def get_ol_model_fn(
    cfg: OLPretrainConfig, inputs: TrainingInputs
) -> Callable[[AFADataset], pl.LightningModule]:
    def f(dataset: AFADataset) -> pl.LightningModule:
        n_features = dataset.feature_shape.numel()
        n_classes = dataset.label_shape.numel()
        _features, labels = dataset.get_all_data()
        class_probabilities = get_class_frequencies(labels)

        n_selections = inputs.unmasker().get_n_selections(
            dataset.feature_shape
        )
        pq_module = OLPQModule(
            n_features=n_features,
            n_classes=n_classes,
            n_actions=n_selections + 1,
            cfg=cfg.pq_module,
        )
        lit_model = LitOLPQModule(
            pq_module=pq_module,
            class_probabilities=class_probabilities,
            n_feature_dims=len(dataset.feature_shape),
            min_masking_probability=cfg.min_masking_probability,
            max_masking_probability=cfg.max_masking_probability,
            lr=cfg.lr,
        )
        return lit_model

    return f


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain_model/ol",
    config_name="config",
)
def main(cfg: OLPretrainConfig) -> None:
    cfg = cast("OLPretrainConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")
    cfg = replace(
        cfg,
        supervised_learning=limit_supervised_learning(
            cfg.supervised_learning, smoke_test=cfg.smoke_test
        ),
    )

    with training_run(cfg, "pretraining", tags=["ol"], config=cfg):
        inputs = load_inputs(cfg)
        model_bundle = supervised_learning(
            train_dataset=inputs.train_dataset(),
            val_dataset=inputs.val_dataset(),
            cfg=cfg.supervised_learning,
            model_fn=get_ol_model_fn(cfg=cfg, inputs=inputs),
            metric_to_monitor="val_loss_many_observations",
            monitor_mode="min",
            use_wandb=cfg.use_wandb,
            device=cfg.device,
        )
        save_result(model_bundle, cfg, cfg, stage="pretraining")


if __name__ == "__main__":
    main()
