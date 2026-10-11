from collections.abc import Callable
from typing import cast

import hydra
import lightning as pl
from omegaconf import OmegaConf
from torchrl.modules import MLP

from afabench.components.methods.rl.common.utils import (
    str_to_activation_class_mapping,
)
from afabench.components.methods.rl.odin.config import ODINPretrainConfig
from afabench.components.methods.rl.odin.models import (
    ODINPretrainingModel,
    PartialVAE,
    PointNet,
    PointNetType,
)
from afabench.core.types import AFADataset
from afabench.core.utils import get_class_frequencies
from afabench.fit.inputs import FitInputs
from afabench.fit.supervised_learning import supervised_pretraining_main


def get_odin_model_fn(
    cfg: ODINPretrainConfig, _inputs: FitInputs
) -> Callable[[AFADataset], pl.LightningModule]:
    def f(
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
            latent_size=cfg.partial_vae.latent_size,
        )
        model = ODINPretrainingModel(
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

    return f


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/pretrain_model/odin",
    config_name="config",
)
def main(cfg: ODINPretrainConfig) -> None:
    cfg = cast("ODINPretrainConfig", OmegaConf.to_object(cfg))
    supervised_pretraining_main(cfg, tags=["odin"], model_fn=get_odin_model_fn)


if __name__ == "__main__":
    main()
