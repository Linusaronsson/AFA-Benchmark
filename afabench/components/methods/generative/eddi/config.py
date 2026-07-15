from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig

cs = ConfigStore.instance()


@dataclass
class EDDIPointNetConfig:
    identity_size: int = 20
    identity_network_num_cells: list[int] = field(
        default_factory=lambda: [20, 20]
    )
    output_size: int = 40
    feature_map_encoder_num_cells: list[int] = field(
        default_factory=lambda: [500]
    )
    max_embedding_norm: float = 1.0


@dataclass
class EDDIPartialVAEConfig:
    lr: float = 1e-3
    patience: int = 5
    encoder_num_cells: list[int] = field(
        default_factory=lambda: [500, 500, 200]
    )
    latent_size: int = 20
    kl_scaling_factor: float = 0.1
    decoder_num_cells: list[int] = field(
        default_factory=lambda: [200, 500, 500]
    )


@dataclass
class EDDIClassifierConfig:
    lr: float = 1e-3
    num_cells: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.3
    patience: int = 5
    classifier_loss_scaling_factor: float = 1.0


@dataclass
class EDDIPretrainingConfig:
    dataset_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    seed: int = 42
    device: str = "cuda"
    n_annealing_epochs: int = 1
    start_kl_scaling_factor: float = 0.1
    end_kl_scaling_factor: float = 0.1
    min_mask: float = 0.1
    max_mask: float = 0.9
    epochs: int = 1000

    pointnet: EDDIPointNetConfig = field(default_factory=EDDIPointNetConfig)
    partial_vae: EDDIPartialVAEConfig = field(
        default_factory=EDDIPartialVAEConfig
    )
    classifier: EDDIClassifierConfig = field(
        default_factory=EDDIClassifierConfig
    )


cs.store(name="pretrain_eddi", node=EDDIPretrainingConfig)


@dataclass
class EDDITrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    pretrained_model_bundle_path: str
    classifier_bundle_path: str
    save_path: str

    hard_budget: int | None
    soft_budget_param: float | None
    device: str | None
    seed: int | None

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    use_wandb: bool
    smoke_test: bool


cs.store(name="train_eddi", node=EDDITrainingConfig)
