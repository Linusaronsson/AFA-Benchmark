"""CLI for completing fixed training views with an ODIN PVAE."""

from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.missing_values.config import RestoreTrainingViewsConfig
from afabench.missing_values.restoration import restore_and_save_training_views


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/missing_values",
    config_name="restore_training_views",
)
def main(cfg: RestoreTrainingViewsConfig) -> None:
    restore_and_save_training_views(
        cast("RestoreTrainingViewsConfig", OmegaConf.to_object(cfg))
    )


if __name__ == "__main__":
    main()
