from pathlib import Path
from typing import TYPE_CHECKING, cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import afabench.components.methods.rl.odin.config  # noqa: F401

if TYPE_CHECKING:
    from afabench.components.methods.rl.odin.config import ODINTrainConfig


def test_odin_train_config_composes_for_workflow_overrides() -> None:
    config_dir = Path("extra/conf/scripts/train_method/odin").resolve()

    with initialize_config_dir(
        version_base=None,
        config_dir=str(config_dir),
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                "train_dataset_bundle_path=train.bundle",
                "val_dataset_bundle_path=val.bundle",
                "pretrained_model_bundle_path=model.bundle",
                "save_path=method.bundle",
                "components/initializers@initializer=cold",
                "components/unmaskers@unmasker=direct",
                "hard_budget=null",
                "soft_budget_param=0.15",
                "experiment@_global_=cube_without_noise",
                "additional_generation_fraction=1.0",
            ],
        )

    train_cfg = cast("ODINTrainConfig", OmegaConf.to_object(cfg))

    assert not hasattr(train_cfg.agent, "replay_buffer_batch_size")
