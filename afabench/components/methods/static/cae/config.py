from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.components.methods.static.common.config import (
    StaticClassifierConfig,
    StaticSelectorConfig,
)
from afabench.training.contract import TrainingContract, store_contract_config

cs = ConfigStore.instance()


@dataclass
class CAEArchitectureConfig:
    selector: StaticSelectorConfig
    classifier: StaticClassifierConfig


@dataclass
class CAETabularArchitectureConfig(CAEArchitectureConfig):
    pass  # no additional fields


@dataclass
class CAEImageArchitectureConfig(CAEArchitectureConfig):
    backbone_type: str
    image_size: int
    patch_size: int


cs.store(
    group="components/cae_architecture",
    name="tabular",
    node=CAETabularArchitectureConfig,
)
cs.store(
    group="components/cae_architecture",
    name="image",
    node=CAEImageArchitectureConfig,
)


@dataclass(frozen=True, kw_only=True)
class CAETrainingConfig(TrainingContract):
    batch_size: int

    architecture: CAEArchitectureConfig


store_contract_config(name="train_cae", config_class=CAETrainingConfig)
