from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.gdfs.config import (
    GDFSImageArchitectureConfig,
    GDFSTrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.train.image import (
    train_image,
)
from afabench.components.methods.discriminative.gdfs.train.tabular import (
    train_tabular,
)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/gdfs",
    config_name="config",
)
def main(cfg: GDFSTrainingConfig) -> None:
    cfg = cast("GDFSTrainingConfig", OmegaConf.to_object(cfg))
    if isinstance(cfg.architecture, GDFSImageArchitectureConfig):
        train_image(cfg)
    else:
        train_tabular(cfg)


if __name__ == "__main__":
    main()
