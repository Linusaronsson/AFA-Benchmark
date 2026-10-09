import logging
from dataclasses import replace
from typing import cast

import hydra
import torch
from omegaconf.omegaconf import OmegaConf

from afabench.components.methods.rl.common.training import limit_training_loop
from afabench.components.methods.rl.jafa.config import JAFATrainConfig
from afabench.components.methods.rl.jafa.training import train_jafa
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/jafa",
    config_name="config",
)
def main(cfg: JAFATrainConfig) -> None:
    cfg = cast("JAFATrainConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")
    cfg = replace(
        cfg,
        rl_training_loop=limit_training_loop(
            cfg.rl_training_loop, smoke_test=cfg.smoke_test
        ),
    )

    with fit_run(cfg, tags=["jafa"], config=cfg) as logger:
        afa_method = train_jafa(cfg, load_inputs(cfg), logger)
        save_result(afa_method, cfg, cfg)


if __name__ == "__main__":
    main()
