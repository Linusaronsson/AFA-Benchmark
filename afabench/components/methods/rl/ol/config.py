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
class OLPQModuleConfig:
    n_hiddens: list[int]
    p_dropout: float
    use_feature_mask: bool


@dataclass(frozen=True, kw_only=True)
class OLPretrainConfig(PretrainingContract):
    supervised_learning: SupervisedLearningConfig

    min_masking_probability: float
    max_masking_probability: float
    lr: float
    pq_module: OLPQModuleConfig


store_contract_config(name="pretrain_ol", config_class=OLPretrainConfig)


@dataclass
class OLAgentConfig:
    eps_init: float
    eps_end: float
    eps_annealing_fraction: float

    replay_buffer_batch_size: int
    replay_buffer_size: int

    num_epochs: int
    max_grad_norm: float
    lr: float
    update_tau: float

    loss_function: str
    delay_value: bool
    double_dqn: bool

    gamma: float
    lmbda: float


@dataclass(frozen=True, kw_only=True)
class OLTrainConfig(TrainingContract):
    mdp: AFAMDPConfig
    rl_training_loop: AFARLTrainingLoopConfig
    agent: OLAgentConfig
    pretrained_model_lr: float
    activate_joint_training_after_fraction: float
    replay_buffer_device_same_as_device: bool

    reward_method: str
    mcdrop_samples: int


store_contract_config(name="train_ol", config_class=OLTrainConfig)
