from dataclasses import replace
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.l2m.config import (
    L2MTrainingConfig,
)
from afabench.components.methods.discriminative.l2m.train import train_l2m
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/l2m",
    config_name="config",
)
def main(cfg: L2MTrainingConfig) -> None:
    cfg = cast("L2MTrainingConfig", OmegaConf.to_object(cfg))
    if cfg.smoke_test:
        # Model width comes from the pretrained model, so the smoke test
        # runs the real heads on short sequences and a small context set.
        cfg = replace(
            cfg,
            sequence_length=16,
            n_steps=2,
            checkpoint_interval=1,
            n_validation_tasks=2,
            context_set_size=8,
        )
    with fit_run(cfg, tags=["l2m"], config=cfg) as metric_logger:
        inputs = load_inputs(cfg)
        result = train_l2m(cfg, metric_logger=metric_logger.log, inputs=inputs)
        save_result(result, cfg, cfg)


if __name__ == "__main__":
    main()
