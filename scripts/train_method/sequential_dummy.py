import logging
from typing import cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.components.methods.dummy.config import (
    SequentialDummyTrainConfig,
)
from afabench.components.methods.dummy.train import train_sequential_dummy
from afabench.training.inputs import load_inputs
from afabench.training.run import save_result, training_run

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/sequential_dummy",
    config_name="config",
)
def main(cfg: SequentialDummyTrainConfig) -> None:
    cfg = cast("SequentialDummyTrainConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    torch.set_float32_matmul_precision("medium")

    inputs = load_inputs(cfg)
    with training_run(cfg, "training", tags=["sequential_dummy"], config=cfg):
        afa_method = train_sequential_dummy(cfg, inputs)
        save_result(afa_method, cfg, cfg, stage="training")


if __name__ == "__main__":
    main()
