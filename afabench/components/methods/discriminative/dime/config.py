from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig

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


@dataclass
class DIMEPretrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str
    save_path: str

    batch_size: int
    seed: int | None
    device: str | None
    lr: float
    nepochs: int
    patience: int
    min_masking_probability: float
    max_masking_probability: float

    architecture: DIMEArchitectureConfig

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool


cs.store(name="pretrain_dime", node=DIMEPretrainingConfig)


@dataclass
class DIMETrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str

    batch_size: int
    lr: float
    hard_budget: int | None
    soft_budget_param: float | None
    nepochs: int
    patience: int
    eps: float
    eps_decay: float
    eps_steps: int
    device: str | None
    seed: int | None

    architecture: DIMEArchitectureConfig

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool

    min_lr: float | None = None


cs.store(name="train_dime", node=DIMETrainingConfig)
