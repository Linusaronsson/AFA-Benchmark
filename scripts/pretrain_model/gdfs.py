from dataclasses import replace
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.gdfs.config import (
    GDFSImageArchitectureConfig,
    GDFSPretrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.pretrain.image import (
    pretrain_image,
)
from afabench.components.methods.discriminative.gdfs.pretrain.tabular import (
    pretrain_tabular,
)
from afabench.training.inputs import load_inputs
from afabench.training.run import save_result, training_run


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain_model/gdfs",
    config_name="config",
)
def main(cfg: GDFSPretrainingConfig) -> None:
    cfg = cast("GDFSPretrainingConfig", OmegaConf.to_object(cfg))
    if cfg.smoke_test:
        cfg = replace(cfg, nepochs=1, patience=1)
    with training_run(
        cfg, "pretraining", tags=["gdfs"], config=cfg
    ) as metric_logger:
        inputs = load_inputs(cfg)
        if isinstance(cfg.architecture, GDFSImageArchitectureConfig):
            result = pretrain_image(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        else:
            result = pretrain_tabular(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        save_result(result, cfg, cfg, stage="pretraining")


if __name__ == "__main__":
    main()
