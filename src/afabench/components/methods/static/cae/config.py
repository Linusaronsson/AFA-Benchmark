from dataclasses import dataclass

from afabench.components.methods.common.architecture import ArchitectureKind
from afabench.components.methods.static.common.config import (
    StaticClassifierConfig,
    StaticSelectorConfig,
)
from afabench.fit.contract import TrainingContract, store_contract_config


@dataclass
class CAEImageArchitectureConfig:
    backbone_type: str
    image_size: int
    patch_size: int


@dataclass(frozen=True, kw_only=True)
class CAETrainingConfig(TrainingContract):
    batch_size: int

    selector: StaticSelectorConfig
    classifier: StaticClassifierConfig

    # The tabular architecture is the selector and classifier MLPs alone
    architecture_kind: ArchitectureKind
    image_architecture: CAEImageArchitectureConfig


store_contract_config(name="train_cae", config_class=CAETrainingConfig)
