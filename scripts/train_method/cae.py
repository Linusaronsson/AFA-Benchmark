from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.common.architecture import ArchitectureKind
from afabench.components.methods.static.cae.config import (
    CAETrainingConfig,
)
from afabench.components.methods.static.cae.train.image import train_image
from afabench.components.methods.static.cae.train.tabular import (
    train_tabular,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result
from afabench.fit.smoke_test import limit


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/cae",
    config_name="config",
)
def main(cfg: CAETrainingConfig) -> None:
    cfg = cast("CAETrainingConfig", OmegaConf.to_object(cfg))
    cfg = limit(
        cfg,
        selector={"nepochs": 1, "patience": 1},
        classifier={"nepochs": 1},
    )
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["cae"], inputs=inputs) as metric_logger:
        if cfg.architecture_kind is ArchitectureKind.IMAGE:
            result = train_image(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        else:
            result = train_tabular(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        save_result(result, cfg, inputs=inputs)


if __name__ == "__main__":
    main()
