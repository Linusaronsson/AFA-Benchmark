from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.static.common.config import (
    StaticClassifierConfig,
    StaticSelectorConfig,
)
from afabench.components.unmaskers.config import UnmaskerConfig

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


@dataclass
class CAETrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str
    save_path: str

    batch_size: int
    hard_budget: int | None
    soft_budget_param: float | None
    device: str | None
    seed: int | None

    architecture: CAEArchitectureConfig

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool


cs.store(name="train_cae", node=CAETrainingConfig)
