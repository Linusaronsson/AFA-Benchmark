from pathlib import Path
from typing import cast

import hydra

from afabench.common.bundle import load_bundle
from afabench.common.config_classes import (
    CAETraining2DConfig,
    CAETrainingConfig,
)
from afabench.static.train.cae_image import train_image
from afabench.static.train.cae_tabular import train_tabular

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
