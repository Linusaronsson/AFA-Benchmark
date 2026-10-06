"""
The training contract: what the pipeline passes to every script.

See `docs/adr/0001-training-contract-as-library.md`. Method configs inherit
`TrainingContract` or `PretrainingContract` and add their own fields.
"""

from dataclasses import dataclass
from typing import Literal

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig

type TrainingStage = Literal["pretraining", "training"]


@dataclass(frozen=True, kw_only=True)
class PretrainingContract:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str
    save_path: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    dataset_key: str
    device: str
    seed: int
    use_wandb: bool = False
    smoke_test: bool = False


@dataclass(frozen=True, kw_only=True)
class TrainingContract(PretrainingContract):
    pretrained_model_bundle_path: str | None = None
    hard_budget: int | None
    soft_budget_param: float | None


def store_contract_config(
    *, name: str, config_class: type[PretrainingContract]
) -> None:
    """
    Register a config class inheriting a contract in Hydra's ConfigStore.

    OmegaConf marks a frozen dataclass read-only, which would reject YAML
    values and command-line overrides, so the stored node is made writable.
    `OmegaConf.to_object` still returns the frozen dataclass.
    """
    node = OmegaConf.structured(config_class)
    OmegaConf.set_readonly(node, False)
    ConfigStore.instance().store(name=name, node=node)
