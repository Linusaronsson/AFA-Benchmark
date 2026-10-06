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
class ODINPointNetConfig:
    type: str
    identity_size: int
    max_embedding_norm: float
    output_size: int
    feature_map_encoder_num_cells: list[int]
    feature_map_encoder_activation_class: str
    feature_map_encoder_dropout: float


@dataclass
class ODINEncoderConfig:
    num_cells: list[int]
    activation_class: str
    dropout: float


@dataclass
class ODINPartialVAEConfig:
    latent_size: int
    decoder_num_cells: list[int]
    decoder_activation_class: str
    decoder_dropout: float


@dataclass
class ODINClassifierConfig:
    num_cells: list[int]
    activation_class: str
    dropout: float


@dataclass(frozen=True, kw_only=True)
class ODINPretrainConfig(PretrainingContract):
    supervised_learning: SupervisedLearningConfig

    min_masking_probability: float
    max_masking_probability: float
    lr: float
    start_kl_scaling_factor: float
    end_kl_scaling_factor: float
    n_annealing_epoch_fraction: float
    classifier_loss_scaling_factor: float
    pointnet: ODINPointNetConfig
    encoder: ODINEncoderConfig
    partial_vae: ODINPartialVAEConfig
    classifier: ODINClassifierConfig


store_contract_config(name="pretrain_odin", config_class=ODINPretrainConfig)


@dataclass
class ODINAgentConfig:
    gamma: float
    lmbda: float

    clip_epsilon: float
    entropy_bonus: bool
    entropy_coef: float
    critic_coef: float
    loss_critic_type: str

    num_epochs: int
    lr: float
    max_grad_norm: float

    value_num_cells: list[int]
    value_dropout: float
    policy_num_cells: list[int]
    policy_dropout: float


@dataclass(frozen=True, kw_only=True)
class ODINTrainConfig(TrainingContract):
    mdp: AFAMDPConfig
    rl_training_loop: AFARLTrainingLoopConfig
    agent: ODINAgentConfig
    additional_generation_fraction: float
    generation_batch_size: int


store_contract_config(name="train_odin", config_class=ODINTrainConfig)
