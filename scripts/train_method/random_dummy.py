import logging
from typing import cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.components.methods.dummy.config import RandomDummyTrainConfig
from afabench.components.methods.dummy.train import train_random_dummy
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/random_dummy",
    config_name="config",
)
def main(cfg: RandomDummyTrainConfig) -> None:
    cfg = cast("RandomDummyTrainConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")

    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["random_dummy"], config=cfg):
        afa_method = train_random_dummy(cfg, inputs)
        save_result(afa_method, cfg, cfg)


if __name__ == "__main__":
    main()
