from dataclasses import asdict
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
from afabench.core.utils import initialize_wandb_run


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/cae",
    config_name="config",
)
def main(cfg: CAETrainingConfig) -> None:
    cfg = cast("CAETrainingConfig", OmegaConf.to_object(cfg))
    wandb_run = None
    if cfg.use_wandb:
        wandb_run = initialize_wandb_run(
            cfg=asdict(cfg),
            job_type="training",
            tags=["cae"],
        )

    if isinstance(cfg.architecture, CAEImageArchitectureConfig):
        train_image(
            cfg,
            metric_logger=wandb_run.log if wandb_run is not None else None,
        )
    else:
        train_tabular(
            cfg,
            metric_logger=wandb_run.log if wandb_run is not None else None,
        )


if __name__ == "__main__":
    main()
