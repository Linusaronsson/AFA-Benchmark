from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig

cs = ConfigStore.instance()


@dataclass
class RandomDummyTrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str | None
    save_path: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    hard_budget: int | None
    soft_budget_param: float | None

    device: str
    seed: int | None
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="train_random_dummy", node=RandomDummyTrainConfig)


@dataclass
class SequentialDummyTrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str | None
    save_path: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    hard_budget: int | None
    soft_budget_param: float | None

    device: str
    seed: int | None
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="train_sequential_dummy", node=SequentialDummyTrainConfig)
