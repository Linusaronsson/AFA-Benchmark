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


@dataclass
class ODINPretrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    classifier_bundle_path: str | None
    save_path: str
    device: str

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
    unmasker: UnmaskerConfig
    seed: int | None = None
    use_wandb: bool = False
    smoke_test: bool = False
    initializer: InitializerConfig | None = None


cs.store(name="pretrain_odin", node=ODINPretrainConfig)


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
    replay_buffer_batch_size: int

    value_num_cells: list[int]
    value_dropout: float
    policy_num_cells: list[int]
    policy_dropout: float


@dataclass
class ODINTrainConfig:
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
    agent: ODINAgentConfig
    additional_generation_fraction: float
    generation_batch_size: int
    seed: int | None = None
    use_wandb: bool = False
    smoke_test: bool = False
    device: str | None = None

    hard_budget: int | None = None


cs.store(name="train_odin", node=ODINTrainConfig)
