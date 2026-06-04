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


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain_model/gdfs",
    config_name="config",
)
def main(cfg: GDFSPretrainingConfig) -> None:
    cfg = cast("GDFSPretrainingConfig", OmegaConf.to_object(cfg))
    if isinstance(cfg.architecture, GDFSImageArchitectureConfig):
        pretrain_image(cfg)
    else:
        pretrain_tabular(cfg)


if __name__ == "__main__":
    main()
