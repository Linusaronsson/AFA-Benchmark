from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.oracle.aaco.config import AACOPretrainConfig
from afabench.components.methods.oracle.aaco.train import run
from afabench.fit.run import fit_run, save_result


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/pretrain_model/aaco",
    config_name="config",
)
def main(cfg: AACOPretrainConfig) -> None:
    cfg = cast("AACOPretrainConfig", OmegaConf.to_object(cfg))
    with fit_run(cfg, tags=["aaco"], config=cfg):
        method = run(cfg, hard_budget=None, soft_budget_param=None)
        save_result(method, cfg, cfg)


if __name__ == "__main__":
    main()
