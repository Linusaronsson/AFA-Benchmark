import logging
from typing import cast

import hydra
import torch
from omegaconf.omegaconf import OmegaConf

from afabench.components.methods.rl.common.training import limit_training_loop
from afabench.components.methods.rl.ol.config import OLTrainConfig
from afabench.components.methods.rl.ol.training import train_ol
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/ol",
    config_name="config",
)
def main(cfg: OLTrainConfig) -> None:
    cfg = cast("OLTrainConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")
    cfg = limit_training_loop(cfg)

    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["ol"], inputs=inputs) as logger:
        afa_method = train_ol(cfg, inputs, logger)
        save_result(afa_method, cfg, inputs=inputs)


if __name__ == "__main__":
    main()
