from dataclasses import dataclass

from afabench.training.contract import TrainingContract, store_contract_config


@dataclass(frozen=True, kw_only=True)
class RandomDummyTrainConfig(TrainingContract):
    """The random dummy method has no hyperparameters beyond the contract."""


store_contract_config(
    name="train_random_dummy", config_class=RandomDummyTrainConfig
)


@dataclass(frozen=True, kw_only=True)
class SequentialDummyTrainConfig(TrainingContract):
    """The sequential dummy method has no hyperparameters beyond the contract."""


store_contract_config(
    name="train_sequential_dummy", config_class=SequentialDummyTrainConfig
)
