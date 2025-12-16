import logging
from pathlib import Path

import hydra
import lightning as pl
import torch
from omegaconf import OmegaConf
from torchrl.modules import MLP

from afabench.afa_rl.utils import (
    str_to_activation_class_mapping,
)
from afabench.afa_rl.zannone2019.models import (
    PartialVAE,
    PointNet,
    PointNetType,
    Zannone2019PretrainingModel,
)
from afabench.common.config_classes import Zannone2019PretrainConfig
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
    config_path="../../extra/conf/scripts/pretrain/zannone2019",
    config_name="config",
)
def main(cfg: Zannone2019PretrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    if cfg.use_wandb:
        _run = initialize_wandb_run(
            cfg=cfg, job_type="pretraining", tags=["zannone2019"]
        )

    # If smoke test, override some options
    if cfg.smoke_test:
        log.info("Smoke test detected.")
        cfg.supervised_learning.max_epochs = 1
        cfg.supervised_learning.limit_train_batches = 2
        cfg.supervised_learning.limit_val_batches = 2

    def get_zannone2019_model(
        dataset: AFADataset,
    ) -> pl.LightningModule:
        n_features = dataset.feature_shape.numel()
        n_classes = dataset.label_shape.numel()
        _features, labels = dataset.get_all_data()
        class_probabilities = get_class_frequencies(labels)
        # PointNet or PointNetPlus
        if cfg.pointnet.type == "pointnet":
            pointnet_type = PointNetType.POINTNET
            feature_map_encoder_input_size = cfg.pointnet.identity_size + 1
        elif cfg.pointnet.type == "pointnetplus":
            pointnet_type = PointNetType.POINTNETPLUS
            feature_map_encoder_input_size = cfg.pointnet.identity_size
        else:
            msg = f"PointNet type {
                cfg.pointnet.type
            } not supported. Use 'pointnet' or 'pointnetplus'."
            raise ValueError(msg)

        pointnet = PointNet(
            identity_size=cfg.pointnet.identity_size,
            n_features=n_features + n_classes,
            max_embedding_norm=cfg.pointnet.max_embedding_norm,
            feature_map_encoder=MLP(
                in_features=feature_map_encoder_input_size,
                out_features=cfg.pointnet.output_size,
                num_cells=cfg.pointnet.feature_map_encoder_num_cells,
                dropout=cfg.pointnet.feature_map_encoder_dropout,
                activation_class=str_to_activation_class_mapping[
                    cfg.pointnet.feature_map_encoder_activation_class
                ],
            ),
            pointnet_type=pointnet_type,
        )
        encoder = MLP(
            in_features=cfg.pointnet.output_size,
            out_features=2 * cfg.partial_vae.latent_size,
            num_cells=cfg.encoder.num_cells,
            dropout=cfg.encoder.dropout,
            activation_class=str_to_activation_class_mapping[
                cfg.encoder.activation_class
            ],
        )
        partial_vae = PartialVAE(
            pointnet=pointnet,
            encoder=encoder,
            decoder=MLP(
                in_features=cfg.partial_vae.latent_size,
                out_features=n_features,
                num_cells=cfg.partial_vae.decoder_num_cells,
                dropout=cfg.partial_vae.decoder_dropout,
                activation_class=str_to_activation_class_mapping[
                    cfg.partial_vae.decoder_activation_class
                ],
            ),
        )
        model = Zannone2019PretrainingModel(
            partial_vae=partial_vae,
            # Classifier acts on latent space
            classifier=MLP(
                in_features=cfg.partial_vae.latent_size,
                out_features=n_classes,
                num_cells=cfg.classifier.num_cells,
                dropout=cfg.classifier.dropout,
                activation_class=str_to_activation_class_mapping[
                    cfg.classifier.activation_class
                ],
            ),
            lr=cfg.lr,
            min_masking_probability=cfg.min_masking_probability,
            max_masking_probability=cfg.max_masking_probability,
            class_probabilities=class_probabilities,
            start_kl_scaling_factor=cfg.start_kl_scaling_factor,
            end_kl_scaling_factor=cfg.end_kl_scaling_factor,
            n_annealing_epochs=int(
                cfg.supervised_learning.max_epochs
                * cfg.n_annealing_epoch_fraction
            ),
            classifier_loss_scaling_factor=cfg.classifier_loss_scaling_factor,
        )
        return model

    supervised_learning(
        train_dataset_bundle_path=Path(cfg.train_dataset_bundle_path),
        val_dataset_bundle_path=Path(cfg.val_dataset_bundle_path),
        save_path=Path(cfg.save_path),
        cfg=cfg.supervised_learning,
        get_model=get_zannone2019_model,
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
