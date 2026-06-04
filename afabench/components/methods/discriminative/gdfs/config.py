from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig

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


@dataclass
class GDFSPretrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str
    save_path: str

    batch_size: int
    seed: int
    device: str
    lr: float
    nepochs: int
    patience: int
    min_masking_probability: float
    max_masking_probability: float

    architecture: GDFSArchitectureConfig

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool


cs.store(name="pretrain_gdfs", node=GDFSPretrainingConfig)


@dataclass
class GDFSTrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str

    batch_size: int
    lr: float
    hard_budget: int
    soft_budget_param: float | None
    nepochs: int
    patience: int
    device: str
    seed: int

    architecture: GDFSArchitectureConfig

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool

    min_lr: float | None = None


cs.store(name="train_gdfs", node=GDFSTrainingConfig)
