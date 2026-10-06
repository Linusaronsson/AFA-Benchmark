from dataclasses import dataclass

from afabench.components.methods.rl.common.config import (
    AFAMDPConfig,
    AFARLTrainingLoopConfig,
)
from afabench.training.config import SupervisedLearningConfig
from afabench.training.contract import (
    PretrainingContract,
    TrainingContract,
    store_contract_config,
)


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


@dataclass(frozen=True, kw_only=True)
class JAFAPretrainConfig(PretrainingContract):
    supervised_learning: SupervisedLearningConfig

    min_masking_probability: float
    max_masking_probability: float
    lr: float
    encoder: JAFAEncoderConfig
    classifier: JAFAClassifierConfig


store_contract_config(name="pretrain_jafa", config_class=JAFAPretrainConfig)


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


@dataclass(frozen=True, kw_only=True)
class JAFATrainConfig(TrainingContract):
    mdp: AFAMDPConfig
    rl_training_loop: AFARLTrainingLoopConfig
    agent: JAFAAgentConfig
    pretrained_model_lr: float
    activate_joint_training_after_fraction: float


store_contract_config(name="train_jafa", config_class=JAFATrainConfig)
