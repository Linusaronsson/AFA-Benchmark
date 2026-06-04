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


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/dime",
    config_name="config",
)
def main(cfg: DIMETrainingConfig) -> None:
    cfg = cast("DIMETrainingConfig", OmegaConf.to_object(cfg))
    if isinstance(cfg.architecture, DIMEImageArchitectureConfig):
        train_image(cfg)
    else:
        train_tabular(cfg)


if __name__ == "__main__":
    main()
