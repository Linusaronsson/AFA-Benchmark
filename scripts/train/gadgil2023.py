from pathlib import Path
from typing import cast

import hydra

from afabench.components.methods.discriminative.gdfs.train.image import (
    train_image,
)
from afabench.components.methods.discriminative.gdfs.train.tabular import (
    train_tabular,
)
from afabench.core.bundle import load_bundle
from afabench.core.config_classes import (
    Gadgil2023Training2DConfig,
    Gadgil2023TrainingConfig,
)

IMAGE_DATASET_CLASSNAMES = {
    "ImagenetteDataset",
}


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/gadgil2023",
    config_name="config",
)
def main(cfg: Gadgil2023TrainingConfig | Gadgil2023Training2DConfig) -> None:
    _, manifest = load_bundle(Path(cfg.train_dataset_bundle_path))
    cls = manifest.get("class_name", "")

    if cls in IMAGE_DATASET_CLASSNAMES:
        train_image(cast("Gadgil2023Training2DConfig", cfg))
    else:
        train_tabular(cast("Gadgil2023TrainingConfig", cfg))


if __name__ == "__main__":
    main()
