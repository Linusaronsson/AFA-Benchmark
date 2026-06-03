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
class PermutationTrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str
    save_path: str

    batch_size: int
    hard_budget: int
    soft_budget_param: float | None
    device: str
    seed: int

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    selector: StaticSelectorConfig
    classifier: StaticClassifierConfig

    use_wandb: bool
    smoke_test: bool


cs.store(name="train_permutation", node=PermutationTrainingConfig)
