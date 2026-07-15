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
class OLPQModuleConfig:
    n_hiddens: list[int]
    p_dropout: float
    use_feature_mask: bool


@dataclass
class OLPretrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str | None
    save_path: str
    unmasker: UnmaskerConfig
    device: str

    supervised_learning: SupervisedLearningConfig

    min_masking_probability: float
    max_masking_probability: float
    lr: float
    pq_module: OLPQModuleConfig
    seed: int | None
    use_wandb: bool
    smoke_test: bool
    initializer: InitializerConfig | None = None


cs.store(name="pretrain_ol", node=OLPretrainConfig)


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


@dataclass
class OLTrainConfig:
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
    agent: OLAgentConfig
    pretrained_model_lr: float
    activate_joint_training_after_fraction: float
    seed: int | None
    use_wandb: bool
    smoke_test: bool
    device: str | None
    replay_buffer_device_same_as_device: bool

    reward_method: str
    mcdrop_samples: int

    hard_budget: int | None = None


cs.store(name="train_ol", node=OLTrainConfig)
