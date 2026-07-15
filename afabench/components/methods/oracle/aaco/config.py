from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig

cs = ConfigStore.instance()


@dataclass
class AACOConfig:
    k_neighbors: int = 5
    acquisition_cost: float = 0.05
    hide_val: float = 0.0
    evaluate_final_performance: bool = True
    eval_only_n_samples: int | None = None
    missingness_objective: str = "support_aware"
    dr_min_propensity: float = 1e-3
    dr_max_weight: float | None = 20.0


@dataclass
class AACOTrainConfig:
    aco: AACOConfig
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    save_path: Path
    train_dataset_bundle_path: Path | None = None
    val_dataset_bundle_path: Path | None = None
    dataset_artifact_name: Path | None = None
    classifier_bundle_path: Path | None = None
    seed: int = 42
    device: str = "cpu"
    use_wandb: bool = False
    soft_budget_param: float | None = None
    hard_budget: int | None = None
    experiment_id: str | None = None
    initializer_type: str = "aaco"
    unmasker_type: str = "one_based_index"
    smoke_test: bool = False


cs.store(name="train_aaco", node=AACOTrainConfig)


@dataclass
class AACONNTrainConfig:
    """Config for AACO+NN (behavioral cloning) training."""

    classifier_bundle_path: Path
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    save_path: Path
    aaco_bundle_path: Path | None = None
    pretrained_model_bundle_path: Path | None = None
    train_dataset_bundle_path: Path | None = None
    val_dataset_bundle_path: Path | None = None
    dataset_artifact_name: Path | None = None
    seed: int = 42
    device: str = "cpu"
    max_acquisitions: int | None = None
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    dropout: float = 0.1
    batch_size: int = 256
    max_epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    val_split: float = 0.1
    hard_budget: int | None = None
    soft_budget_param: float | None = None
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="train_aaco_nn", node=AACONNTrainConfig)
