from dataclasses import dataclass

from afabench.components.methods.common.architecture import ArchitectureKind
from afabench.fit.contract import (
    PretrainingContract,
    TrainingContract,
    store_contract_config,
)


@dataclass
class GDFSTabularArchitectureConfig:
    activation: str
    hidden_units: list[int]
    dropout: float


@dataclass
class GDFSImageArchitectureConfig:
    backbone_type: str
    image_size: int
    patch_size: int


@dataclass(frozen=True, kw_only=True)
class GDFSPretrainingConfig(PretrainingContract):
    batch_size: int
    lr: float
    nepochs: int
    patience: int
    min_masking_probability: float
    max_masking_probability: float

    architecture_kind: ArchitectureKind
    tabular_architecture: GDFSTabularArchitectureConfig
    image_architecture: GDFSImageArchitectureConfig


store_contract_config(name="pretrain_gdfs", config_class=GDFSPretrainingConfig)


@dataclass(frozen=True, kw_only=True)
class GDFSTrainingConfig(TrainingContract):
    batch_size: int
    lr: float
    nepochs: int
    patience: int

    # The pretrained model decides which of the two the policy uses
    tabular_architecture: GDFSTabularArchitectureConfig
    image_architecture: GDFSImageArchitectureConfig

    min_lr: float | None = None


store_contract_config(name="train_gdfs", config_class=GDFSTrainingConfig)
