from dataclasses import replace
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.gdfs.config import (
    GDFSImageArchitectureConfig,
    GDFSTrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.train.image import (
    train_image,
)
from afabench.components.methods.discriminative.gdfs.train.tabular import (
    train_tabular,
)
from afabench.training.inputs import load_inputs
from afabench.training.run import save_result, training_run


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/gdfs",
    config_name="config",
)
def main(cfg: GDFSTrainingConfig) -> None:
    cfg = cast("GDFSTrainingConfig", OmegaConf.to_object(cfg))
    if cfg.smoke_test:
        cfg = replace(cfg, nepochs=1, patience=1)
    with training_run(
        cfg, "training", tags=["gdfs"], config=cfg
    ) as metric_logger:
        inputs = load_inputs(cfg)
        inputs.initializer().set_seed(cfg.seed)
        inputs.unmasker().set_seed(cfg.seed)
        if isinstance(cfg.architecture, GDFSImageArchitectureConfig):
            result = train_image(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        else:
            result = train_tabular(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        save_result(result, cfg, cfg, stage="training")


if __name__ == "__main__":
    main()
