from dataclasses import dataclass, field
from pathlib import Path

from hydra.core.config_store import ConfigStore


@dataclass
class TrainingMissingnessConfig:
    """Configuration for a fitted training-data missingness mechanism."""

    mechanism: str = "none"
    p: float = 0.0
    p_obs: float = 0.3
    p_params: float = 0.3
    exclude_inputs: bool = True


@dataclass
class MaterializeTrainingViewsConfig:
    train_dataset_bundle_path: Path
    val_dataset_bundle_path: Path
    train_save_path: Path
    val_save_path: Path
    strategy: str = "restricted"
    seed: int = 0
    missingness: TrainingMissingnessConfig = field(
        default_factory=TrainingMissingnessConfig
    )


@dataclass
class RestoreTrainingViewsConfig:
    train_view_bundle_path: Path
    val_view_bundle_path: Path
    pvae_bundle_path: Path
    train_save_path: Path
    val_save_path: Path
    strategy: str
    seed: int = 0
    batch_size: int = 1024
    device: str = "cpu"
    reference_train_dataset_bundle_path: Path | None = None
    reference_val_dataset_bundle_path: Path | None = None


ConfigStore.instance().store(
    group="components/training_missingness",
    name="none",
    node=TrainingMissingnessConfig,
)
ConfigStore.instance().store(
    name="materialize_training_views_schema",
    node=MaterializeTrainingViewsConfig,
)
ConfigStore.instance().store(
    name="restore_training_views_schema",
    node=RestoreTrainingViewsConfig,
)
