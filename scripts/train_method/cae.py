from dataclasses import replace
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.static.cae.config import (
    CAEImageArchitectureConfig,
    CAETrainingConfig,
)
from afabench.components.methods.static.cae.train.image import train_image
from afabench.components.methods.static.cae.train.tabular import (
    train_tabular,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/cae",
    config_name="config",
)
def main(cfg: CAETrainingConfig) -> None:
    cfg = cast("CAETrainingConfig", OmegaConf.to_object(cfg))
    if cfg.smoke_test:
        cfg = replace(
            cfg,
            architecture=replace(
                cfg.architecture,
                selector=replace(
                    cfg.architecture.selector, nepochs=1, patience=1
                ),
                classifier=replace(cfg.architecture.classifier, nepochs=1),
            ),
        )
    with fit_run(cfg, tags=["cae"], config=cfg) as metric_logger:
        inputs = load_inputs(cfg)
        if isinstance(cfg.architecture, CAEImageArchitectureConfig):
            result = train_image(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        else:
            result = train_tabular(
                cfg, metric_logger=metric_logger.log, inputs=inputs
            )
        save_result(result, cfg, cfg)


if __name__ == "__main__":
    main()
