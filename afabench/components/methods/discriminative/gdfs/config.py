from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.fit.contract import (
    PretrainingContract,
    TrainingContract,
    store_contract_config,
)

cs = ConfigStore.instance()


@dataclass
class GDFSArchitectureConfig:
    pass  # base marker


@dataclass
class GDFSTabularArchitectureConfig(GDFSArchitectureConfig):
    activation: str
    hidden_units: list[int]
    dropout: float


@dataclass
class GDFSImageArchitectureConfig(GDFSArchitectureConfig):
    backbone_type: str
    image_size: int
    patch_size: int


cs.store(
    group="components/gdfs_architecture",
    name="tabular",
    node=GDFSTabularArchitectureConfig,
)
cs.store(
    group="components/gdfs_architecture",
    name="image",
    node=GDFSImageArchitectureConfig,
)


@dataclass(frozen=True, kw_only=True)
class GDFSPretrainingConfig(PretrainingContract):
    batch_size: int
    lr: float
    nepochs: int
    patience: int
    min_masking_probability: float
    max_masking_probability: float

    architecture: GDFSArchitectureConfig


store_contract_config(name="pretrain_gdfs", config_class=GDFSPretrainingConfig)


@dataclass(frozen=True, kw_only=True)
class GDFSTrainingConfig(TrainingContract):
    batch_size: int
    lr: float
    nepochs: int
    patience: int

    architecture: GDFSArchitectureConfig

    min_lr: float | None = None


store_contract_config(name="train_gdfs", config_class=GDFSTrainingConfig)
