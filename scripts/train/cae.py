import hydra
from pathlib import Path
from typing import cast

from afabench.common.config_classes import CAETrainingConfig, CAETraining2DConfig
from afabench.common.bundle import load_bundle

from afabench.static.train.cae_tabular import train_tabular
from afabench.static.train.cae_image import train_image

IMAGE_DATASET_CLASSNAMES = {
    "ImagenetteDataset",
}


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/cae",
    config_name="config",
)
def main(cfg: CAETrainingConfig | CAETraining2DConfig):
    _, manifest = load_bundle(Path(cfg.train_dataset_bundle_path))
    cls = manifest.get("class_name", "")

    if cls in IMAGE_DATASET_CLASSNAMES:
        train_image(cast(CAETraining2DConfig, cfg))
    else:
        train_tabular(cast(CAETrainingConfig, cfg))

if __name__ == "__main__":
    main()
