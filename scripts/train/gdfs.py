from pathlib import Path
from typing import cast

import hydra

from afabench.components.methods.discriminative.gdfs.train.image import (
    train_image,
)
from afabench.components.methods.discriminative.gdfs.train.tabular import (
    train_tabular,
)
from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.config_classes import (
    GDFSTraining2DConfig,
    GDFSTrainingConfig,
)

IMAGE_DATASET_CLASSNAMES = {
    "ImagenetteDataset",
}


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/gdfs",
    config_name="config",
)
def main(cfg: GDFSTrainingConfig | GDFSTraining2DConfig) -> None:
    _, manifest = load_bundle(Path(cfg.train_dataset_bundle_path))
    cls = manifest.get("class_name", "")

    if cls in IMAGE_DATASET_CLASSNAMES:
        train_image(cast("GDFSTraining2DConfig", cfg))
    else:
        train_tabular(cast("GDFSTrainingConfig", cfg))


if __name__ == "__main__":
    main()
