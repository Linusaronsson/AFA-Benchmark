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


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/cae",
    config_name="config",
)
def main(cfg: CAETrainingConfig) -> None:
    cfg = cast("CAETrainingConfig", OmegaConf.to_object(cfg))
    if isinstance(cfg.architecture, CAEImageArchitectureConfig):
        train_image(cfg)
    else:
        train_tabular(cfg)


if __name__ == "__main__":
    main()
