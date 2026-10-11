from collections.abc import Callable
from typing import cast

import hydra
import lightning as pl
from omegaconf import OmegaConf

from afabench.components.methods.rl.ol.config import OLPretrainConfig
from afabench.components.methods.rl.ol.models import (
    LitOLPQModule,
    OLPQModule,
)
from afabench.core.types import AFADataset
from afabench.core.utils import get_class_frequencies
from afabench.fit.inputs import FitInputs
from afabench.fit.supervised_learning import supervised_pretraining_main


def get_ol_model_fn(
    cfg: OLPretrainConfig, inputs: FitInputs
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
    config_path="../../conf/scripts/pretrain_model/ol",
    config_name="config",
)
def main(cfg: OLPretrainConfig) -> None:
    cfg = cast("OLPretrainConfig", OmegaConf.to_object(cfg))
    supervised_pretraining_main(cfg, tags=["ol"], model_fn=get_ol_model_fn)


if __name__ == "__main__":
    main()
