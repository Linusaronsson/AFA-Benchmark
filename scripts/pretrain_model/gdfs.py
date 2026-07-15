from dataclasses import asdict
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.gdfs.config import (
    GDFSImageArchitectureConfig,
    GDFSPretrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.pretrain.image import (
    pretrain_image,
)
from afabench.components.methods.discriminative.gdfs.pretrain.tabular import (
    pretrain_tabular,
)
from afabench.core.utils import initialize_wandb_run


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain_model/gdfs",
    config_name="config",
)
def main(cfg: GDFSPretrainingConfig) -> None:
    cfg = cast("GDFSPretrainingConfig", OmegaConf.to_object(cfg))
    wandb_run = None
    if cfg.use_wandb:
        wandb_run = initialize_wandb_run(
            cfg=asdict(cfg),
            job_type="pretraining",
            tags=["gdfs"],
        )

    if isinstance(cfg.architecture, GDFSImageArchitectureConfig):
        pretrain_image(
            cfg,
            metric_logger=wandb_run.log if wandb_run is not None else None,
        )
    else:
        pretrain_tabular(
            cfg,
            metric_logger=wandb_run.log if wandb_run is not None else None,
        )


if __name__ == "__main__":
    main()
