from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig

cs = ConfigStore.instance()


@dataclass
class TrainMaskedMLPClassifierConfig:
    train_dataset_path: str
    val_dataset_path: str
    save_path: str

    epochs: int
    batch_size: int
    limit_train_batches: int | None
    limit_val_batches: int | None
    min_masking_probability: float
    max_masking_probability: float

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    lr: float
    num_cells: list[int]
    dropout: float

    seed: int
    device: str
    use_wandb: bool = False
    smoke_test: bool = False
    eval_only_n_samples: int | None = None


cs.store(
    name="train_masked_mlp_classifier", node=TrainMaskedMLPClassifierConfig
)


@dataclass
class TrainMaskedViTClassifierConfig:
    train_dataset_path: str
    val_dataset_path: str
    save_path: str

    batch_size: int
    epochs: int
    min_masking_probability: float
    max_masking_probability: float

    model_name: str
    image_size: int
    patch_size: int
    patience: int
    min_lr: float

    lr: float
    seed: int
    device: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool


cs.store(
    name="train_masked_vit_classifier", node=TrainMaskedViTClassifierConfig
)
