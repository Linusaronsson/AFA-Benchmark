from dataclasses import dataclass

from afabench.fit.contract import PretrainingContract


@dataclass
class SupervisedLearningConfig:
    batch_size: int
    max_epochs: int
    checkpoint_earliest_batch: int
    early_stopping_min_batches: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    val_check_interval: int
    limit_train_batches: int | None
    limit_val_batches: int | None


@dataclass(frozen=True, kw_only=True)
class SupervisedPretrainingConfig(PretrainingContract):
    """A pretraining config that `supervised_pretraining_main` can run."""

    supervised_learning: SupervisedLearningConfig
