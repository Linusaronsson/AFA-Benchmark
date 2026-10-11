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
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result
from afabench.fit.smoke_test import limit


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/pretrain_model/gdfs",
    config_name="config",
)
def main(cfg: GDFSPretrainingConfig) -> None:
    cfg = cast("GDFSPretrainingConfig", OmegaConf.to_object(cfg))
    cfg = limit(cfg, nepochs=1, patience=1)
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["gdfs"], inputs=inputs) as metric_logger:
        if isinstance(cfg.architecture, GDFSImageArchitectureConfig):
            result = pretrain_image(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        else:
            result = pretrain_tabular(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        save_result(result, cfg, inputs=inputs)


if __name__ == "__main__":
    main()
