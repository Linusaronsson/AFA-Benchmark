import logging

import hydra

from afabench.common.config_classes import AACOTrainConfig
from scripts.train.aaco import run

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/train/aaco",
    config_name="config",
)
def main(cfg: AACOTrainConfig) -> None:
    logger.debug(cfg)
    run(cfg)


if __name__ == "__main__":
    main()
