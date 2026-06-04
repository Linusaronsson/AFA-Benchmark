from pathlib import Path
from typing import cast

import hydra

from afabench.components.methods.discriminative.dime.config import (
    DIMETraining2DConfig,
    DIMETrainingConfig,
)
from afabench.components.methods.discriminative.dime.train.image import (
    train_image,
)
from afabench.components.methods.discriminative.dime.train.tabular import (
    train_tabular,
)
from afabench.core.bundle_system.bundle import load_bundle

IMAGE_DATASET_CLASSNAMES = {
    "ImagenetteDataset",
}


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/dime",
    config_name="config",
)
def main(cfg: DIMETrainingConfig | DIMETraining2DConfig) -> None:
    _, manifest = load_bundle(Path(cfg.train_dataset_bundle_path))
    cls = manifest.get("class_name", "")

    if cls in IMAGE_DATASET_CLASSNAMES:
        train_image(cast("DIMETraining2DConfig", cfg))
    else:
        train_tabular(cast("DIMETrainingConfig", cfg))


if __name__ == "__main__":
    main()
