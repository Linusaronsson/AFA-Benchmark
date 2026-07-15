"""CLI for producing fixed train/validation views under missingness."""

from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.missing_values.config import MaterializeTrainingViewsConfig
from afabench.missing_values.views import save_materialized_training_views


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/missing_values",
    config_name="materialize_training_views",
)
def main(cfg: MaterializeTrainingViewsConfig) -> None:
    save_materialized_training_views(
        cast("MaterializeTrainingViewsConfig", OmegaConf.to_object(cfg))
    )


if __name__ == "__main__":
    main()
