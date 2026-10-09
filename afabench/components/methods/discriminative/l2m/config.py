from dataclasses import dataclass

from afabench.components.methods.discriminative.l2m.task_sampler import (
    FeatureSource,
)
from afabench.fit.contract import (
    PretrainingContract,
    TrainingContract,
    store_contract_config,
)


def _require_a_validation_task(n_validation_tasks: int) -> None:
    # The checkpoint is chosen by the loss on the held-out validation tasks.
    if n_validation_tasks < 1:
        msg = f"n_validation_tasks={n_validation_tasks} must be at least 1"
        raise ValueError(msg)


# Not frozen: OmegaConf would make the nested node read-only.
@dataclass(kw_only=True)
class L2MArchitectureConfig:
    """Keyword arguments of `L2MModel` other than the dataset's shapes."""

    model_dim: int
    embedding_depth: int
    n_layers: int
    n_heads: int
    feedforward_dim: int


@dataclass(frozen=True, kw_only=True)
class L2MPretrainingConfig(PretrainingContract):
    # See `sample_task`.
    feature_source: FeatureSource
    architecture: L2MArchitectureConfig
    # Tasks per step, and instances per task.
    batch_size: int
    sequence_length: int
    n_steps: int
    lr: float
    warmup_steps: int
    checkpoint_interval: int
    n_validation_tasks: int
    missingness_cap: float

    def __post_init__(self) -> None:
        _require_a_validation_task(self.n_validation_tasks)


store_contract_config(name="pretrain_l2m", config_class=L2MPretrainingConfig)


@dataclass(frozen=True, kw_only=True)
class L2MTrainingConfig(TrainingContract):
    # The same feature source as the pretrained model's, see `sample_task`.
    feature_source: FeatureSource
    # Tasks per step, and instances per task.
    batch_size: int
    sequence_length: int
    n_steps: int
    # Policy head learning rate, and the lower one of the backbone and the
    # built-in classifier it is fine-tuned with.
    policy_lr: float
    backbone_lr: float
    # Fixed straight-through Gumbel-softmax temperature.
    temperature: float
    checkpoint_interval: int
    n_validation_tasks: int
    missingness_cap: float
    # Validation-split instances stored in the method bundle.
    context_set_size: int

    def __post_init__(self) -> None:
        _require_a_validation_task(self.n_validation_tasks)


store_contract_config(name="train_l2m", config_class=L2MTrainingConfig)
