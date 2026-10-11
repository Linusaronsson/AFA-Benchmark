from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.components.methods.discriminative.gdfs.config import (
    GDFSTrainingConfig,
)
from afabench.components.methods.discriminative.gdfs.train import train
from afabench.fit.inputs import load_inputs
from afabench.fit.run import fit_run, save_result
from afabench.fit.smoke_test import limit


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/train_method/gdfs",
    config_name="config",
)
def main(cfg: GDFSTrainingConfig) -> None:
    cfg = cast("GDFSTrainingConfig", OmegaConf.to_object(cfg))
    cfg = limit(cfg, nepochs=1, patience=1)
    inputs = load_inputs(cfg)
    with fit_run(cfg, tags=["gdfs"], inputs=inputs) as metric_logger:
        result = train(cfg, metric_logger=metric_logger.log, inputs=inputs)
        save_result(result, cfg, inputs=inputs)


if __name__ == "__main__":
    main()
