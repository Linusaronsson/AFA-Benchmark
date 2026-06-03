from pathlib import Path
from typing import cast

import hydra

from afabench.components.methods.static.cae.train.image import train_image
from afabench.components.methods.static.cae.train.tabular import (
    train_tabular,
)
from afabench.core.bundle import load_bundle
from afabench.core.config_classes import (
    CAETraining2DConfig,
    CAETrainingConfig,
)

IMAGE_DATASET_CLASSNAMES = {
    "ImagenetteDataset",
}


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/cae",
    config_name="config",
)
def main(cfg: CAETrainingConfig | CAETraining2DConfig) -> None:
    _, manifest = load_bundle(Path(cfg.train_dataset_bundle_path))
    cls = manifest.get("class_name", "")

    if cls in IMAGE_DATASET_CLASSNAMES:
        train_image(cast("CAETraining2DConfig", cfg))
    else:
        train_tabular(cast("CAETrainingConfig", cfg))


if __name__ == "__main__":
    main()
