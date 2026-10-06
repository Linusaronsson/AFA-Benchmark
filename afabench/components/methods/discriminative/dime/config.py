from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.training.contract import (
    PretrainingContract,
    TrainingContract,
    store_contract_config,
)

cs = ConfigStore.instance()


@dataclass
class DIMEArchitectureConfig:
    pass  # base marker


@dataclass
class DIMETabularArchitectureConfig(DIMEArchitectureConfig):
    activation: str
    hidden_units: list[int]
    dropout: float


@dataclass
class DIMEImageArchitectureConfig(DIMEArchitectureConfig):
    backbone_type: str
    image_size: int
    patch_size: int


cs.store(
    group="components/dime_architecture",
    name="tabular",
    node=DIMETabularArchitectureConfig,
)
cs.store(
    group="components/dime_architecture",
    name="image",
    node=DIMEImageArchitectureConfig,
)


@dataclass(frozen=True, kw_only=True)
class DIMEPretrainingConfig(PretrainingContract):
    batch_size: int
    lr: float
    nepochs: int
    patience: int
    min_masking_probability: float
    max_masking_probability: float

    architecture: DIMEArchitectureConfig


store_contract_config(name="pretrain_dime", config_class=DIMEPretrainingConfig)


@dataclass(frozen=True, kw_only=True)
class DIMETrainingConfig(TrainingContract):
    batch_size: int
    lr: float
    nepochs: int
    patience: int
    eps: float
    eps_decay: float
    eps_steps: int

    architecture: DIMEArchitectureConfig

    min_lr: float | None = None


store_contract_config(name="train_dime", config_class=DIMETrainingConfig)
