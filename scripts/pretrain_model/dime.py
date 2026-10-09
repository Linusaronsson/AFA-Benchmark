from dataclasses import replace
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.dime.config import (
    DIMEImageArchitectureConfig,
    DIMEPretrainingConfig,
)
from afabench.components.methods.discriminative.dime.pretrain.image import (
    pretrain_image,
)
from afabench.components.methods.discriminative.dime.pretrain.tabular import (
    pretrain_tabular,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/pretrain_model/dime",
    config_name="config",
)
def main(cfg: DIMEPretrainingConfig) -> None:
    cfg = cast("DIMEPretrainingConfig", OmegaConf.to_object(cfg))
    if cfg.smoke_test:
        cfg = replace(cfg, nepochs=1, patience=1)
    with fit_run(cfg, tags=["dime"], config=cfg) as metric_logger:
        inputs = load_inputs(cfg)
        if isinstance(cfg.architecture, DIMEImageArchitectureConfig):
            result = pretrain_image(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        else:
            result = pretrain_tabular(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        save_result(result, cfg, cfg)


if __name__ == "__main__":
    main()
