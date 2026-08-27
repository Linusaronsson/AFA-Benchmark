from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.rl.common.config import (
    AFAMDPConfig,
    AFARLTrainingLoopConfig,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.training.config import SupervisedLearningConfig

cs = ConfigStore.instance()


@dataclass
class JAFAEncoderConfig:
    output_size: int
    reading_block_cells: list[int]
    writing_block_cells: list[int]
    memory_size: int
    processing_steps: int
    dropout: float


@dataclass
class JAFAClassifierConfig:
    num_cells: list[int]


@dataclass
class JAFAPretrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str | None
    save_path: str
    device: str

    supervised_learning: SupervisedLearningConfig

    min_masking_probability: float
    max_masking_probability: float
    lr: float
    encoder: JAFAEncoderConfig
    classifier: JAFAClassifierConfig
    unmasker: UnmaskerConfig
    seed: int | None = None
    use_wandb: bool = False
    smoke_test: bool = False
    initializer: InitializerConfig | None = None
    respect_source_availability: bool = False


cs.store(name="pretrain_jafa", node=JAFAPretrainConfig)


@dataclass
class JAFAAgentConfig:
    eps_init: float
    eps_end: float
    eps_annealing_fraction: float

    num_epochs: int
    max_grad_norm: float
    lr: float
    update_tau: float

    action_value_num_cells: list[int]
    action_value_dropout: float

    loss_function: str
    delay_value: bool
    double_dqn: bool

    gamma: float
    lmbda: float
    use_action_availability: bool = False


@dataclass
class JAFATrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    pretrained_model_bundle_path: str
    classifier_bundle_path: str | None
    save_path: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    mdp: AFAMDPConfig
    rl_training_loop: AFARLTrainingLoopConfig
    soft_budget_param: float | None
    agent: JAFAAgentConfig
    pretrained_model_lr: float
    activate_joint_training_after_fraction: float
    seed: int | None = None
    use_wandb: bool = False
    smoke_test: bool = False
    device: str | None = None

    hard_budget: int | None = None


cs.store(name="train_jafa", node=JAFATrainConfig)
