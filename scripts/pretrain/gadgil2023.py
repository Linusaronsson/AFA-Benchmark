import hydra
from pathlib import Path
from typing import cast

from afabench.common.config_classes import Gadgil2023PretrainingConfig, Gadgil2023Pretraining2DConfig
from afabench.common.bundle import load_bundle

from afabench.afa_discriminative.pretrain.gadgil2023_tabular import pretrain_tabular
from afabench.afa_discriminative.pretrain.gadgil2023_image import pretrain_image

IMAGE_DATASET_CLASSNAMES = {
    "ImagenetteDataset",
}


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/pretrain/gadgil2023",
    config_name="config",
)
def main(cfg: Gadgil2023PretrainingConfig | Gadgil2023Pretraining2DConfig) -> None:
    _, manifest = load_bundle(Path(cfg.train_dataset_bundle_path))
    cls = manifest.get("class_name", "")

    if cls in IMAGE_DATASET_CLASSNAMES:
        pretrain_image(cast(Gadgil2023Pretraining2DConfig, cfg))
    else:
        pretrain_tabular(cast(Gadgil2023PretrainingConfig, cfg))

if __name__ == "__main__":
    main()
