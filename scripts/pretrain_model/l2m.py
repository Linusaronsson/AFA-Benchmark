from dataclasses import replace
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.l2m.config import (
    L2MPretrainingConfig,
)
from afabench.components.methods.discriminative.l2m.pretrain import (
    pretrain_l2m,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/pretrain_model/l2m",
    config_name="config",
)
def main(cfg: L2MPretrainingConfig) -> None:
    cfg = cast("L2MPretrainingConfig", OmegaConf.to_object(cfg))
    if cfg.smoke_test:
        # Model width is kept, so the smoke test runs the real architecture.
        cfg = replace(
            cfg,
            sequence_length=16,
            n_steps=2,
            checkpoint_interval=1,
            n_validation_tasks=2,
        )
    with fit_run(cfg, tags=["l2m"], config=cfg) as metric_logger:
        inputs = load_inputs(cfg)
        result = pretrain_l2m(
            cfg, metric_logger=metric_logger.log, inputs=inputs
        )
        save_result(result, cfg, cfg)


if __name__ == "__main__":
    main()
