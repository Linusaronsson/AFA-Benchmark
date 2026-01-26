import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as pl
import torch
from omegaconf import OmegaConf

from afabench.afa_rl.kachuee2019.models import (
    Kachuee2019PQModule,
    LitKachuee2019PQModule,
)
from afabench.common.config_classes import Kachuee2019PretrainConfig
from afabench.common.custom_types import AFADataset
from afabench.common.supervised_learning import supervised_learning
from afabench.common.unmaskers.utils import get_afa_unmasker_from_config
from afabench.common.utils import (
    get_class_frequencies,
    initialize_wandb_run,
    set_seed,
)

log = logging.getLogger(__name__)


def get_kachuee2019_model_fn(
    cfg: Kachuee2019PretrainConfig,
) -> Callable[[AFADataset], pl.LightningModule]:
    def f(dataset: AFADataset) -> pl.LightningModule:
        n_features = dataset.feature_shape.numel()
        n_classes = dataset.label_shape.numel()
        _features, labels = dataset.get_all_data()
        class_probabilities = get_class_frequencies(labels)

        n_selections = get_afa_unmasker_from_config(
            cfg.unmasker
        ).get_n_selections(dataset.feature_shape)
        pq_module = Kachuee2019PQModule(
            n_features=n_features,
            n_classes=n_classes,
            n_actions=n_selections + 1,
            cfg=cfg.pq_module,
        )
        lit_model = LitKachuee2019PQModule(
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
    config_path="../../extra/conf/scripts/pretrain/kachuee2019",
    config_name="config",
)
def main(cfg: Kachuee2019PretrainConfig) -> None:
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        _run = initialize_wandb_run(
            cfg=cast(
                "dict[str,Any]", OmegaConf.to_container(cfg, resolve=True)
            ),
            job_type="pretraining",
            tags=["kachuee2019"],
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
        model_fn=get_kachuee2019_model_fn(cfg=cfg),
        metric_to_monitor="val_loss_many_observations",
        monitor_mode="min",
        use_wandb=cfg.use_wandb,
        device=cfg.device,
        metadata_to_save_in_bundle={
            "train_dataset_bundle_path": cfg.train_dataset_bundle_path,
            "seed": cfg.seed,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
    )


if __name__ == "__main__":
    main()
