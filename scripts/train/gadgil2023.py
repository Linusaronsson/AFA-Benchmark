import hydra
from pathlib import Path
from typing import cast

from afabench.common.config_classes import Gadgil2023TrainingConfig, Gadgil2023Training2DConfig
from afabench.common.bundle import load_bundle

from afabench.afa_discriminative.train.gadgil2023_tabular import train_tabular
from afabench.afa_discriminative.train.gadgil2023_image import train_image

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
        train_image(cast(Gadgil2023Training2DConfig, cfg))
    else:
        train_tabular(cast(Gadgil2023TrainingConfig, cfg))

if __name__ == "__main__":
    main()
