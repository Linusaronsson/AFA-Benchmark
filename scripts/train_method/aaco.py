from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.oracle.aaco.config import AACOTrainConfig
from afabench.components.methods.oracle.aaco.train import run
from afabench.training.run import save_result, training_run


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train_method/aaco",
    config_name="config",
)
def main(cfg: AACOTrainConfig) -> None:
    cfg = cast("AACOTrainConfig", OmegaConf.to_object(cfg))
    with training_run(cfg, "training", tags=["aaco"], config=cfg):
        method = run(cfg)
        save_result(method, cfg, cfg, stage="training")


if __name__ == "__main__":
    main()
