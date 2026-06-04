import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
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
from afabench.components.unmaskers.utils import (
    get_afa_unmasker_from_config,
)
from afabench.core.types import AFADataset
from afabench.core.utils import (
    get_class_frequencies,
    initialize_wandb_run,
    set_seed,
)
from afabench.training.supervised_learning import supervised_learning

log = logging.getLogger(__name__)


def get_ol_model_fn(
    cfg: OLPretrainConfig,
) -> Callable[[AFADataset], pl.LightningModule]:
    def f(dataset: AFADataset) -> pl.LightningModule:
        n_features = dataset.feature_shape.numel()
        n_classes = dataset.label_shape.numel()
        _features, labels = dataset.get_all_data()
        class_probabilities = get_class_frequencies(labels)

        n_selections = get_afa_unmasker_from_config(
            cfg.unmasker
        ).get_n_selections(dataset.feature_shape)
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
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        _run = initialize_wandb_run(
            cfg=asdict(cfg),
            job_type="pretraining",
            tags=["ol"],
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
        model_fn=get_ol_model_fn(cfg=cfg),
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
