import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import hydra
import torch
from omegaconf import OmegaConf

from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.config_classes import StopBaselineTrainConfig
from afabench.common.stop_baseline_method import StopBaselineMethod
from afabench.common.utils import set_seed

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from afabench.common.custom_types import AFAClassifier


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/stop_baseline",
    config_name="config",
)
def main(cfg: StopBaselineTrainConfig) -> None:
    log.debug(cfg)
    set_seed(cfg.seed)

    classifier, _classifier_manifest = load_bundle(
        Path(cfg.classifier_bundle_path),
        device=torch.device(cfg.device),
    )
    classifier = cast("AFAClassifier", cast("object", classifier))

    method = StopBaselineMethod(
        afa_classifier=classifier,
        device=torch.device(cfg.device),
    )

    save_bundle(
        obj=method,
        path=Path(cfg.save_path),
        metadata={"config": OmegaConf.to_container(cfg, resolve=True)},
    )
    log.info("Stop baseline method saved to: %s", cfg.save_path)


if __name__ == "__main__":
    main()
