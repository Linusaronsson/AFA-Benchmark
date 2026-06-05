from dataclasses import asdict
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.oracle.aaco.config import AACOTrainConfig
from afabench.components.methods.oracle.aaco.train import run
from afabench.core.utils import initialize_wandb_run


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/aaco",
    config_name="config",
)
def main(cfg: AACOTrainConfig) -> None:
    cfg = cast("AACOTrainConfig", OmegaConf.to_object(cfg))
    if cfg.use_wandb:
        _run = initialize_wandb_run(
            cfg=asdict(cfg),
            job_type="training",
            tags=["aaco"],
        )

    run(cfg)


if __name__ == "__main__":
    main()
