from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.oracle.aaco.config import AACOTrainConfig
from afabench.components.methods.oracle.aaco.train import run
from afabench.fit.run import fit_run, save_result


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/aaco",
    config_name="config",
)
def main(cfg: AACOTrainConfig) -> None:
    cfg = cast("AACOTrainConfig", OmegaConf.to_object(cfg))
    with fit_run(cfg, tags=["aaco"], config=cfg):
        method = run(
            cfg,
            hard_budget=cfg.hard_budget,
            soft_budget_param=cfg.soft_budget_param,
        )
        save_result(method, cfg, cfg)


if __name__ == "__main__":
    main()
