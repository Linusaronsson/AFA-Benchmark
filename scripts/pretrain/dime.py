from pathlib import Path
from typing import cast

import hydra

from afabench.components.methods.discriminative.dime.config import (
    DIMEPretraining2DConfig,
    DIMEPretrainingConfig,
)
from afabench.components.methods.discriminative.dime.pretrain.image import (
    pretrain_image,
)
from afabench.components.methods.discriminative.dime.pretrain.tabular import (
    pretrain_tabular,
)
from afabench.core.bundle_system.bundle import load_bundle

IMAGE_DATASET_CLASSNAMES = {
    "ImagenetteDataset",
}


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain/dime",
    config_name="config",
)
def main(
    cfg: DIMEPretrainingConfig | DIMEPretraining2DConfig,
) -> None:
    _, manifest = load_bundle(Path(cfg.train_dataset_bundle_path))
    cls = manifest.get("class_name", "")

    if cls in IMAGE_DATASET_CLASSNAMES:
        pretrain_image(cast("DIMEPretraining2DConfig", cfg))
    else:
        pretrain_tabular(cast("DIMEPretrainingConfig", cfg))


if __name__ == "__main__":
    main()
