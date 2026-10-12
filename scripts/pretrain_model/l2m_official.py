from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.l2m_official.config import (
    L2MOfficialPretrainingConfig,
)
from afabench.components.methods.discriminative.l2m_official.pretrain import (
    pretrain_l2m_official,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result
from afabench.fit.smoke_test import limit


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/pretrain_model/l2m_official",
    config_name="config",
)
def main(cfg: L2MOfficialPretrainingConfig) -> None:
    cfg = cast("L2MOfficialPretrainingConfig", OmegaConf.to_object(cfg))
    # Model width is kept, so the smoke test runs the real architecture.
    cfg = limit(
        cfg,
        # Every task has 50 context instances, the sampler's own default.
        sequence_length=64,
        pretraining_pool_size=64,
        n_steps=2,
        checkpoint_interval=1,
        n_validation_tasks=2,
    )
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["l2m_official"], inputs=inputs) as metric_logger:
        result = pretrain_l2m_official(
            cfg, metric_logger=metric_logger.log, inputs=inputs
        )
        save_result(result, cfg, inputs=inputs)


if __name__ == "__main__":
    main()
