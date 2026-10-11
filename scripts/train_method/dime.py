from dataclasses import replace
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.dime.config import (
    DIMETrainingConfig,
)
from afabench.components.methods.discriminative.dime.train import train
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
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["dime"], inputs=inputs) as metric_logger:
        inputs.initializer().set_seed(cfg.seed)
        inputs.unmasker().set_seed(cfg.seed)
        result = train(cfg, metric_logger=metric_logger.log, inputs=inputs)
        save_result(result, cfg, inputs=inputs)


if __name__ == "__main__":
    main()
