from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig

cs = ConfigStore.instance()


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
    activation: str
    min_masking_probability: float
    max_masking_probability: float

    hidden_units: list[int]
    dropout: float

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool


cs.store(name="pretrain_gdfs", node=GDFSPretrainingConfig)


@dataclass
class GDFSPretraining2DConfig:
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
    backbone_type: str

    image_size: int
    patch_size: int

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool


cs.store(name="pretrain_gdfs", node=GDFSPretraining2DConfig)


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
    activation: str
    device: str
    seed: int

    hidden_units: list[int]
    dropout: float
    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool


cs.store(name="train_gdfs", node=GDFSTrainingConfig)


@dataclass
class GDFSTraining2DConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str

    batch_size: int
    lr: float
    min_lr: float
    hard_budget: int
    soft_budget_param: float | None
    nepochs: int
    patience: int
    device: str
    seed: int
    backbone_type: str

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool


cs.store(name="train_gdfs", node=GDFSTraining2DConfig)
