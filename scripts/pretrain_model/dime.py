from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.dime.config import (
    DIMEImageArchitectureConfig,
    DIMEPretrainingConfig,
)
from afabench.components.methods.discriminative.dime.pretrain.image import (
    pretrain_image,
)
from afabench.components.methods.discriminative.dime.pretrain.tabular import (
    pretrain_tabular,
)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain_model/dime",
    config_name="config",
)
def main(cfg: DIMEPretrainingConfig) -> None:
    cfg = cast("DIMEPretrainingConfig", OmegaConf.to_object(cfg))
    if isinstance(cfg.architecture, DIMEImageArchitectureConfig):
        pretrain_image(cfg)
    else:
        pretrain_tabular(cfg)


if __name__ == "__main__":
    main()
