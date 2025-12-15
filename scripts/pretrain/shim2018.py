import logging
from pathlib import Path

import hydra
import lightning as pl
import torch
from omegaconf import OmegaConf

from afabench.afa_rl.shim2018.models import (
    LitShim2018EmbedderClassifier,
    ReadProcessEncoder,
    Shim2018Embedder,
    Shim2018MLPClassifier,
)
from afabench.common.config_classes import Shim2018PretrainConfig
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
    config_path="../../extra/conf/scripts/pretrain/shim2018",
    config_name="config",
)
def main(cfg: Shim2018PretrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        _run = initialize_wandb_run(
            cfg=cfg, job_type="pretraining", tags=["shim2018"]
        )

    # If smoke test, override some options
    if cfg.smoke_test:
        log.info("Smoke test detected.")
        cfg.supervised_learning.max_epochs = 1
        cfg.supervised_learning.limit_train_batches = 2
        cfg.supervised_learning.limit_val_batches = 2

    def get_shim2018_model(dataset: AFADataset) -> pl.LightningModule:
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
        embedder = Shim2018Embedder(encoder)
        classifier = Shim2018MLPClassifier(
            cfg.encoder.output_size, n_classes, tuple(cfg.classifier.num_cells)
        )
        lit_model = LitShim2018EmbedderClassifier(
            embedder=embedder,
            classifier=classifier,
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
        get_model=get_shim2018_model,
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
