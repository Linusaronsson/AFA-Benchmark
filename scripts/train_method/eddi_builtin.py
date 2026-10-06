import logging
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.generative.eddi.config import (
    EDDITrainingConfig,
)
from afabench.components.methods.generative.eddi.training import (
    build_eddi_afa_method,
)
from afabench.training.inputs import load_inputs
from afabench.training.run import save_result, training_run

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/eddi_builtin",
    config_name="config",
)
def main(cfg: EDDITrainingConfig) -> None:
    cfg = cast("EDDITrainingConfig", OmegaConf.to_object(cfg))
    log.debug(cfg)
    with training_run(cfg, "training", tags=["eddi_builtin"], config=cfg):
        afa_method = build_eddi_afa_method(
            load_inputs(cfg), classifier_bundle_path=None
        )
        save_result(afa_method, cfg, cfg, stage="training")


if __name__ == "__main__":
    main()
