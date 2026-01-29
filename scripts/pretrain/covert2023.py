from pathlib import Path
from typing import cast

import hydra

from afabench.afa_discriminative.pretrain.covert2023_image import (
    pretrain_image,
)
from afabench.afa_discriminative.pretrain.covert2023_tabular import (
    pretrain_tabular,
)
from afabench.common.bundle import load_bundle
from afabench.common.config_classes import (
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
