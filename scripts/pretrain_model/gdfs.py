from pathlib import Path
from typing import cast

import hydra

from afabench.components.methods.discriminative.gdfs.config import (
    GDFSPretraining2DConfig,
    GDFSPretrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.pretrain.image import (
    pretrain_image,
)
from afabench.components.methods.discriminative.gdfs.pretrain.tabular import (
    pretrain_tabular,
)
from afabench.core.bundle_system.bundle import load_bundle

IMAGE_DATASET_CLASSNAMES = {
    "ImagenetteDataset",
}


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain_model/gdfs",
    config_name="config",
)
def main(
    cfg: GDFSPretrainingConfig | GDFSPretraining2DConfig,
) -> None:
    _, manifest = load_bundle(Path(cfg.train_dataset_bundle_path))
    cls = manifest.get("class_name", "")

    if cls in IMAGE_DATASET_CLASSNAMES:
        pretrain_image(cast("GDFSPretraining2DConfig", cfg))
    else:
        pretrain_tabular(cast("GDFSPretrainingConfig", cfg))


if __name__ == "__main__":
    main()
