from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.l2m_official.config import (
    L2MOfficialTrainingConfig,
)
from afabench.components.methods.discriminative.l2m_official.train import (
    train_l2m_official,
)
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result
from afabench.fit.smoke_test import limit


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/l2m_official",
    config_name="config",
)
def main(cfg: L2MOfficialTrainingConfig) -> None:
    cfg = cast("L2MOfficialTrainingConfig", OmegaConf.to_object(cfg))
    # Model width comes from the pretrained model, so the smoke test runs
    # the real heads on short sequences and a small context set.
    cfg = limit(
        cfg,
        # Every task has 50 context instances, the sampler's own default.
        sequence_length=64,
        pretraining_pool_size=64,
        n_steps=2,
        checkpoint_interval=1,
        n_validation_tasks=2,
        context_set_size=8,
    )
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["l2m_official"], inputs=inputs) as metric_logger:
        result = train_l2m_official(
            cfg, metric_logger=metric_logger.log, inputs=inputs
        )
        save_result(result, cfg, inputs=inputs)


if __name__ == "__main__":
    main()
