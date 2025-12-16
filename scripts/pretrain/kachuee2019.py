import logging
from pathlib import Path

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
from afabench.common.utils import (
    get_class_frequencies,
    initialize_wandb_run,
    set_seed,
)

log = logging.getLogger(__name__)


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
            cfg=cfg, job_type="pretraining", tags=["kachuee2019"]
        )

    # If smoke test, override some options
    if cfg.smoke_test:
        log.info("Smoke test detected.")
        cfg.supervised_learning.max_epochs = 1
        cfg.supervised_learning.limit_train_batches = 2
        cfg.supervised_learning.limit_val_batches = 2

    def get_kachuee2019_model(dataset: AFADataset) -> pl.LightningModule:
        n_features = dataset.feature_shape.numel()
        n_classes = dataset.label_shape.numel()
        _features, labels = dataset.get_all_data()
        class_probabilities = get_class_frequencies(labels)
        pq_module = Kachuee2019PQModule(
            n_features=n_features, n_classes=n_classes, cfg=cfg.pq_module
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

    supervised_learning(
        train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
        val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
        save_path=Path(cfg.save_path),
        cfg=cfg.supervised_learning,
        get_model=get_kachuee2019_model,
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
