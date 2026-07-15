from dataclasses import asdict
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
from afabench.core.utils import initialize_wandb_run


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/dime",
    config_name="config",
)
def main(cfg: DIMETrainingConfig) -> None:
    cfg = cast("DIMETrainingConfig", OmegaConf.to_object(cfg))
    wandb_run = None
    if cfg.use_wandb:
        wandb_run = initialize_wandb_run(
            cfg=asdict(cfg),
            job_type="training",
            tags=["dime"],
        )

    if isinstance(cfg.architecture, DIMEImageArchitectureConfig):
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
