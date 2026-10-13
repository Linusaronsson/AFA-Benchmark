from dataclasses import dataclass

from afabench.components.methods.common.architecture import ArchitectureKind
from afabench.fit.contract import (
    PretrainingContract,
    TrainingContract,
    store_contract_config,
)


@dataclass
class DIMETabularArchitectureConfig:
    activation: str
    hidden_units: list[int]
    dropout: float


@dataclass
class DIMEImageArchitectureConfig:
    backbone_type: str
    image_size: int
    patch_size: int


@dataclass(frozen=True, kw_only=True)
class DIMEPretrainingConfig(PretrainingContract):
    batch_size: int
    lr: float
    nepochs: int
    patience: int
    min_masking_probability: float
    max_masking_probability: float

    architecture_kind: ArchitectureKind
    tabular_architecture: DIMETabularArchitectureConfig
    image_architecture: DIMEImageArchitectureConfig


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

    # The pretrained model decides which of the two the policy uses
    tabular_architecture: DIMETabularArchitectureConfig
    image_architecture: DIMEImageArchitectureConfig

    min_lr: float | None = None


store_contract_config(name="train_dime", config_class=DIMETrainingConfig)
