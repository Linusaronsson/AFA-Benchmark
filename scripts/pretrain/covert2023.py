from pathlib import Path
from typing import cast

import hydra

from afabench.components.methods.discriminative.dime.pretrain.image import (
    pretrain_image,
)
from afabench.components.methods.discriminative.dime.pretrain.tabular import (
    pretrain_tabular,
)
from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.config_classes import (
    Covert2023Pretraining2DConfig,
    Covert2023PretrainingConfig,
)

IMAGE_DATASET_CLASSNAMES = {
    "ImagenetteDataset",
}


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain/covert2023",
    config_name="config",
)
def main(
    cfg: Covert2023PretrainingConfig | Covert2023Pretraining2DConfig,
) -> None:
    _, manifest = load_bundle(Path(cfg.train_dataset_bundle_path))
    cls = manifest.get("class_name", "")

    if cls in IMAGE_DATASET_CLASSNAMES:
        pretrain_image(cast("Covert2023Pretraining2DConfig", cfg))
    else:
        pretrain_tabular(cast("Covert2023PretrainingConfig", cfg))


if __name__ == "__main__":
    main()
