import logging
from pathlib import Path
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.generative.eddi.config import (
    EDDITrainingConfig,
)
from afabench.components.methods.generative.eddi.training import (
    build_eddi_afa_method,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/eddi_external",
    config_name="config",
)
def main(cfg: EDDITrainingConfig) -> None:
    cfg = cast("EDDITrainingConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["eddi_external"], inputs=inputs):
        afa_method = build_eddi_afa_method(
            inputs,
            classifier_bundle_path=Path(cfg.classifier_bundle_path),
        )
        save_result(afa_method, cfg, inputs=inputs)


if __name__ == "__main__":
    main()
