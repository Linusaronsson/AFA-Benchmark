from dataclasses import replace
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.dime.config import (
    DIMEImageArchitectureConfig,
    DIMETrainingConfig,
)
from afabench.components.methods.discriminative.dime.train.image import (
    train_image,
)
from afabench.components.methods.discriminative.dime.train.tabular import (
    train_tabular,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/dime",
    config_name="config",
)
def main(cfg: DIMETrainingConfig) -> None:
    cfg = cast("DIMETrainingConfig", OmegaConf.to_object(cfg))
    if cfg.smoke_test:
        cfg = replace(cfg, nepochs=1, patience=1)
    with fit_run(cfg, tags=["dime"], config=cfg) as metric_logger:
        inputs = load_inputs(cfg)
        inputs.initializer().set_seed(cfg.seed)
        inputs.unmasker().set_seed(cfg.seed)
        if isinstance(cfg.architecture, DIMEImageArchitectureConfig):
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
