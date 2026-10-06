from dataclasses import dataclass, field
from pathlib import Path

from afabench.training.contract import TrainingContract, store_contract_config


@dataclass
class AACOConfig:
    k_neighbors: int = 5
    acquisition_cost: float = 0.05
    hide_val: float = 0.0
    mask_seed: int = 0
    evaluate_final_performance: bool = True
    eval_only_n_samples: int | None = None


@dataclass(frozen=True, kw_only=True)
class AACOTrainConfig(TrainingContract):
    """Shared by the AACO pretraining and training stages."""

    aco: AACOConfig
    dataset_artifact_name: Path | None = None
    experiment_id: str | None = None
    initializer_type: str = "aaco"
    unmasker_type: str = "one_based_index"


store_contract_config(name="train_aaco", config_class=AACOTrainConfig)


@dataclass(frozen=True, kw_only=True)
class AACONNTrainConfig(TrainingContract):
    """Config for AACO+NN (behavioral cloning) training."""

    aaco_bundle_path: Path | None = None
    dataset_artifact_name: Path | None = None
    max_acquisitions: int | None = None
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    dropout: float = 0.1
    batch_size: int = 256
    max_epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    val_split: float = 0.1


store_contract_config(name="train_aaco_nn", config_class=AACONNTrainConfig)
