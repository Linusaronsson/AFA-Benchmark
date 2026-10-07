from dataclasses import FrozenInstanceError, dataclass, fields
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue

from afabench.components.initializers.config import InitializerConfig
from afabench.fit.contract import (
    PretrainingContract,
    TrainingContract,
    store_contract_config,
)

PRETRAINING_CONTRACT_FIELDS = {
    "train_dataset_bundle_path",
    "val_dataset_bundle_path",
    "classifier_bundle_path",
    "save_path",
    "initializer",
    "unmasker",
    "dataset_key",
    "device",
    "seed",
    "use_wandb",
    "smoke_test",
}

TRAINING_CONTRACT_FIELDS = {
    *PRETRAINING_CONTRACT_FIELDS,
    "method_name",
    "pretrained_model_bundle_path",
    "hard_budget",
    "soft_budget_param",
}

CONTRACT_OVERRIDES = [
    "train_dataset_bundle_path=train.bundle",
    "val_dataset_bundle_path=val.bundle",
    "classifier_bundle_path=classifier.bundle",
    "save_path=method.bundle",
    "method_name=my_method",
    "dataset_key=cube_without_noise",
    "hard_budget=5",
    "soft_budget_param=null",
    "device=cpu",
    "seed=3",
]


@dataclass(frozen=True)
class _MethodTrainConfig(TrainingContract):
    learning_rate: float
    n_epochs: int = 2


store_contract_config(
    name="test_training_contract_method", config_class=_MethodTrainConfig
)


def _write_method_config(config_dir: Path) -> None:
    (config_dir / "config.yaml").write_text(
        """\
defaults:
  - test_training_contract_method
  - _self_

learning_rate: 0.1
initializer:
  class_name: RandomInitializer
  kwargs:
    num_initial_features: 0
unmasker:
  class_name: DirectUnmasker
  kwargs: {}
"""
    )


def _compose_method_config(
    config_dir: Path, overrides: list[str]
) -> _MethodTrainConfig:
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="config", overrides=overrides)
    return cast("_MethodTrainConfig", OmegaConf.to_object(cfg))


def test_stage_contracts_have_their_expected_fields() -> None:
    assert {f.name for f in fields(PretrainingContract)} == (
        PRETRAINING_CONTRACT_FIELDS
    )
    assert {f.name for f in fields(TrainingContract)} == (
        TRAINING_CONTRACT_FIELDS
    )


def test_method_config_inheriting_contract_composes_with_hydra_overrides(
    tmp_path: Path,
) -> None:
    _write_method_config(tmp_path)

    cfg = _compose_method_config(
        tmp_path, [*CONTRACT_OVERRIDES, "learning_rate=0.5"]
    )

    assert isinstance(cfg, _MethodTrainConfig)
    assert cfg.save_path == "method.bundle"
    assert cfg.method_name == "my_method"
    assert cfg.dataset_key == "cube_without_noise"
    assert cfg.hard_budget == 5
    assert cfg.soft_budget_param is None
    assert cfg.seed == 3
    assert cfg.initializer == InitializerConfig(
        class_name="RandomInitializer",
        kwargs={"num_initial_features": 0},
    )
    assert cfg.pretrained_model_bundle_path is None
    assert cfg.use_wandb is False
    assert cfg.smoke_test is False
    assert cfg.learning_rate == 0.5
    assert cfg.n_epochs == 2


def test_composed_contract_is_frozen(tmp_path: Path) -> None:
    _write_method_config(tmp_path)
    cfg = _compose_method_config(tmp_path, CONTRACT_OVERRIDES)

    with pytest.raises(FrozenInstanceError):
        cfg.seed = 4  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("field_name", ["seed", "method_name"])
def test_training_contract_requires_its_field(
    tmp_path: Path, field_name: str
) -> None:
    _write_method_config(tmp_path)
    overrides = [
        o for o in CONTRACT_OVERRIDES if not o.startswith(f"{field_name}=")
    ]

    with pytest.raises(MissingMandatoryValue, match=field_name):
        _compose_method_config(tmp_path, overrides)
