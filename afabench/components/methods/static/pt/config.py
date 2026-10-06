from dataclasses import dataclass

from afabench.components.methods.static.common.config import (
    StaticClassifierConfig,
    StaticSelectorConfig,
)
from afabench.training.contract import TrainingContract, store_contract_config


@dataclass(frozen=True, kw_only=True)
class PermutationTrainingConfig(TrainingContract):
    batch_size: int

    selector: StaticSelectorConfig
    classifier: StaticClassifierConfig


store_contract_config(
    name="train_permutation", config_class=PermutationTrainingConfig
)
