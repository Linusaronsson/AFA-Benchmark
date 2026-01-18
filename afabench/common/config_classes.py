from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

# --- DATASETS ---


@dataclass
class SplitRatioConfig:
    train: float  # ratio of training data
    val: float  # ratio of validation data
    test: float  # ratio of test data


@dataclass
class DatasetConfig:
    class_name: str
    kwargs: dict[str, Any]


@dataclass
class DatasetGenerationConfig:
    # where to save the instances. Each instance will be saved in a separate subfolder.
    save_path: str
    instance_indices: list[int]  # we will save multiple instances at once
    seeds: list[int]  # one seed per dataset instance
    split_ratio: SplitRatioConfig  # how to split train/val/test
    dataset: DatasetConfig


cs.store(name="dataset_generation", node=DatasetGenerationConfig)

# -- Unmaskers --


@dataclass
class ImagePatchUnmaskerConfig:
    image_side_length: int
    n_channels: int
    patch_size: int


@dataclass
class UnmaskerConfig:
    class_name: str
    # config: ImagePatchUnmaskerConfig | None
    kwargs: dict[str, Any]


# -- Initializers --


# @dataclass
# class ManualInitializerConfig:
#     flat_feature_indices: list[int]


# @dataclass
# class AACODefaultInitializerConfig:
#     dataset_name: str


# @dataclass
# class FixedRandomInitializerConfig:
#     unmask_ratio: float  # how many features to unmask


# @dataclass
# class RandomPerEpisodeInitializerConfig:
#     unmask_ratio: float  # how many features to unmask


# @dataclass
# class MutualInformationInitializerConfig:
#     unmask_ratio: float  # how many features to unmask


# @dataclass
# class LeastInformativeInitializerConfig:
#     unmask_ratio: float  # how many features to unmask


@dataclass
class InitializerConfig:
    class_name: str
    # config: (
    #     ManualInitializerConfig
    #     | AACODefaultInitializerConfig
    #     | FixedRandomInitializerConfig
    #     | RandomPerEpisodeInitializerConfig
    #     | MutualInformationInitializerConfig
    #     | LeastInformativeInitializerConfig
    #     | None
    # )
    kwargs: dict[str, Any]


# --- Common components


# Common things that we need when doing supervised learning
@dataclass
class SupervisedLearningConfig:
    batch_size: int  # batch size for dataloader
    max_epochs: int
    # at which batch we start to keep track of the best performance. Useful for models such as PVAE for which the loss actually increases at the beginning.
    checkpoint_earliest_batch: int
    # minimum number of batches to process before we allow quitting. Useful because many methods fail to learn at the beginning
    early_stopping_min_batches: int
    early_stopping_patience: int  # early stopping patience
    early_stopping_min_delta: float
    val_check_interval: int  # how often to validate
    limit_train_batches: int | None  # only used for smoke tests
    limit_val_batches: int | None  # only used for smoke tests


# --- PRETRAINING MODELS ---

# shim2018


@dataclass
class Shim2018EncoderConfig:
    output_size: int
    reading_block_cells: list[int]
    writing_block_cells: list[int]
    memory_size: int
    processing_steps: int
    dropout: float


@dataclass
class Shim2018ClassifierConfig:
    num_cells: list[int]


@dataclass
class Shim2018PretrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str
    device: str

    supervised_learning: SupervisedLearningConfig

    min_masking_probability: float
    max_masking_probability: float
    lr: float
    encoder: Shim2018EncoderConfig
    classifier: Shim2018ClassifierConfig
    seed: int | None = None
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="pretrain_shim2018", node=Shim2018PretrainConfig)

# zannone2019


@dataclass
class Zannone2019PointNetConfig:
    type: str  # "pointnet" or "pointnetplus"
    identity_size: int
    max_embedding_norm: float
    output_size: int
    feature_map_encoder_num_cells: list[int]
    feature_map_encoder_activation_class: str
    feature_map_encoder_dropout: float


@dataclass
class Zannone2019EncoderConfig:
    num_cells: list[int]
    activation_class: str
    dropout: float


@dataclass
class Zannone2019PartialVAEConfig:
    latent_size: int
    decoder_num_cells: list[int]
    decoder_activation_class: str
    decoder_dropout: float


@dataclass
class Zannone2019ClassifierConfig:
    num_cells: list[int]
    activation_class: str
    dropout: float


@dataclass
class Zannone2019PretrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str
    device: str

    supervised_learning: SupervisedLearningConfig

    min_masking_probability: float
    max_masking_probability: float
    lr: float
    start_kl_scaling_factor: float
    end_kl_scaling_factor: float
    n_annealing_epoch_fraction: float  # fraction of `epochs`
    classifier_loss_scaling_factor: float
    pointnet: Zannone2019PointNetConfig
    encoder: Zannone2019EncoderConfig
    partial_vae: Zannone2019PartialVAEConfig
    classifier: Zannone2019ClassifierConfig
    seed: int | None = None
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="pretrain_zannone2019", node=Zannone2019PretrainConfig)

# kachuee2019


@dataclass
class Kachuee2019PQModuleConfig:
    n_hiddens: list[
        int
        # hidden layers in P network. The hidden layers of the Q network are calculated from this.
    ]
    p_dropout: float


@dataclass
class Kachuee2019PretrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str
    device: str

    supervised_learning: SupervisedLearningConfig

    min_masking_probability: float
    max_masking_probability: float
    lr: float
    pq_module: Kachuee2019PQModuleConfig
    seed: int | None = None
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="pretrain_kachuee2019", node=Kachuee2019PretrainConfig)


# --- TRAINING METHODS ---


@dataclass
class AFARLTrainingLoopConfig:
    # how many frames each batch will contain, *including* frames from all parallel agents
    frames_per_batch: int
    n_batches: int  # for how many batches to train
    # how many steps the agent can take in eval env before episode is terminated,
    eval_max_steps: int
    n_eval_episodes: int  # how many rollouts to average over during evaluation
    device: str | None = None
    env_seed: int | None = None
    eval_n_times: int = (
        10  # how many times to evaluate the agent during the run
    )


@dataclass
class AFAMDPConfig:
    hard_budget: int | None = (
        # how many selections are allowed before episode ends. If None, end when all selections are performed.
        None
    )
    # If False, the agent can choose to terminate the episode without reaching the hard budget. Only relevant when hard_budget is set.
    force_hard_budget: bool = True
    n_agents: int = 1  # how many agents to train in parallel


# shim2018


@dataclass
class Shim2018AgentConfig:
    # epsilon-greedy parameters
    eps_init: float
    eps_end: float
    eps_annealing_fraction: (
        float  # fraction of total number of expected batches
    )

    # Optimization parameters
    num_epochs: int  # how many times to pass over the batch of data received
    max_grad_norm: float
    lr: float
    update_tau: float

    # Module parameters
    action_value_num_cells: list[int]
    action_value_dropout: float

    # Loss parameters
    loss_function: str
    delay_value: bool
    double_dqn: bool

    # Value estimator parameters
    gamma: float
    lmbda: float


@dataclass
class Shim2018TrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    mdp: AFAMDPConfig
    rl_training_loop: AFARLTrainingLoopConfig
    soft_budget_param: float | None
    agent: Shim2018AgentConfig
    pretrained_model_lr: float
    activate_joint_training_after_fraction: float
    seed: int | None = None
    use_wandb: bool = False
    smoke_test: bool = False
    device: str | None = None

    # Aliases, only because snakefile assumes a flat interface
    hard_budget: int | None = None


cs.store(name="train_shim2018", node=Shim2018TrainConfig)

# ma2018


@dataclass
class Ma2018PointNetConfig:
    identity_size: int = 20
    identity_network_num_cells: list[int] = field(
        default_factory=lambda: [20, 20]
    )
    output_size: int = 40
    feature_map_encoder_num_cells: list[int] = field(
        default_factory=lambda: [500]
    )
    max_embedding_norm: float = 1.0


@dataclass
class Ma2018PartialVAEConfig:
    lr: float = 1e-3
    patience: int = 5
    encoder_num_cells: list[int] = field(
        default_factory=lambda: [500, 500, 200]
    )
    latent_size: int = 20
    kl_scaling_factor: float = 0.1
    decoder_num_cells: list[int] = field(
        default_factory=lambda: [200, 500, 500]
    )


@dataclass
class Ma2018ClassifierConfig:
    lr: float = 1e-3
    num_cells: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.3
    patience: int = 5
    classifier_loss_scaling_factor: float = 1.0


@dataclass
class Ma2018PretrainingConfig:
    dataset_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    seed: int = 42
    device: str = "cuda"
    n_annealing_epochs: int = 1
    start_kl_scaling_factor: float = 0.1
    end_kl_scaling_factor: float = 0.1
    min_mask: float = 0.1
    max_mask: float = 0.9
    epochs: int = 1000

    pointnet: Ma2018PointNetConfig = field(
        default_factory=Ma2018PointNetConfig
    )
    partial_vae: Ma2018PartialVAEConfig = field(
        default_factory=Ma2018PartialVAEConfig
    )
    classifier: Ma2018ClassifierConfig = field(
        default_factory=Ma2018ClassifierConfig
    )


cs.store(name="pretrain_ma2018", node=Ma2018PretrainingConfig)


@dataclass
class Ma2018TrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str

    hard_budget: int
    device: str
    seed: int

    initializer: InitializerConfig
    unmasker: UnmaskerConfig


cs.store(name="train_ma2018", node=Ma2018TrainingConfig)


@dataclass
class Covert2023PretrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str

    batch_size: int
    seed: int
    device: str
    lr: float
    nepochs: int
    patience: int
    activation: str
    min_masking_probability: float
    max_masking_probability: float

    hidden_units: list[int]
    dropout: float


cs.store(name="pretrain_covert2023", node=Covert2023PretrainingConfig)


@dataclass
class Covert2023Pretraining2DConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str

    batch_size: int
    seed: int
    device: str
    lr: float
    nepochs: int
    patience: int
    min_masking_probability: float
    max_masking_probability: float

    image_size: int
    patch_size: int


cs.store(name="pretrain_covert2023", node=Covert2023Pretraining2DConfig)


@dataclass
class Covert2023TrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str
    batch_size: int
    lr: float
    hard_budget: int
    nepochs: int
    patience: int
    activation: str
    device: str
    seed: int

    hidden_units: list[int]
    dropout: float
    initializer: InitializerConfig
    unmasker: UnmaskerConfig


cs.store(name="train_covert2023", node=Covert2023TrainingConfig)


@dataclass
class Covert2023Training2DConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str

    batch_size: int
    lr: float
    min_lr: float
    hard_budget: int
    nepochs: int
    patience: int
    device: str
    seed: int

    initializer: InitializerConfig
    unmasker: UnmaskerConfig


cs.store(name="train_covert2023", node=Covert2023Training2DConfig)


@dataclass
class Gadgil2023PretrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str

    batch_size: int
    seed: int
    device: str
    lr: float
    nepochs: int
    patience: int
    activation: str
    min_masking_probability: float
    max_masking_probability: float

    hidden_units: list[int]
    dropout: float


cs.store(name="pretrain_gadgil2023", node=Gadgil2023PretrainingConfig)


@dataclass
class Gadgil2023Pretraining2DConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str

    batch_size: int
    seed: int
    device: str
    lr: float
    nepochs: int
    patience: int
    min_masking_probability: float
    max_masking_probability: float

    image_size: int
    patch_size: int


cs.store(name="pretrain_gadgil2023", node=Gadgil2023Pretraining2DConfig)


@dataclass
class Gadgil2023TrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str

    batch_size: int
    lr: float
    hard_budget: int
    nepochs: int
    patience: int
    activation: str
    eps: float
    eps_decay: float
    eps_steps: int
    device: str
    seed: int

    hidden_units: list[int]
    dropout: float
    initializer: InitializerConfig
    unmasker: UnmaskerConfig


cs.store(name="train_gadgil2023", node=Gadgil2023TrainingConfig)


@dataclass
class Gadgil2023Training2DConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str

    batch_size: int
    lr: float
    min_lr: float
    hard_budget: int
    nepochs: int
    patience: int
    eps: float
    eps_decay: float
    eps_steps: int
    device: str
    seed: int

    initializer: InitializerConfig
    unmasker: UnmaskerConfig


cs.store(name="train_gadgil2023", node=Gadgil2023Training2DConfig)


@dataclass
class StaticSelectorConfig:
    lr: float
    nepochs: int
    num_cells: list[int]
    patience: int


@dataclass
class StaticClassifierConfig:
    lr: float
    nepochs: int
    num_cells: list[int]


@dataclass
class CAETrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str

    batch_size: int
    hard_budget: int
    device: str
    seed: int

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    selector: StaticSelectorConfig
    classifier: StaticClassifierConfig


cs.store(name="train_cae", node=CAETrainingConfig)


@dataclass
class CAETraining2DConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str

    batch_size: int
    image_size: int
    patch_size: int
    hard_budget: int
    device: str
    seed: int

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    selector: StaticSelectorConfig
    classifier: StaticClassifierConfig


cs.store(name="train_cae", node=CAETraining2DConfig)


@dataclass
class PermutationTrainingConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str

    batch_size: int
    hard_budget: int
    device: str
    seed: int

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    selector: StaticSelectorConfig
    classifier: StaticClassifierConfig


cs.store(name="train_permutation", node=PermutationTrainingConfig)


@dataclass
class ResaveConfig:
    trained_model_bundle_path: str
    save_path: str

    device: str
    soft_budget_param: float


cs.store(name="resave", node=ResaveConfig)

# random_dummy


@dataclass
class RandomDummyTrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    hard_budget: int | None  # not used, but pretend that it is
    soft_budget_param: float | None

    device: str
    seed: int | None
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="train_random_dummy", node=RandomDummyTrainConfig)

# sequential_dummy


@dataclass
class SequentialDummyTrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    hard_budget: int | None  # not used, but pretend that it is
    soft_budget_param: float | None

    device: str
    seed: int | None
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="train_sequential_dummy", node=SequentialDummyTrainConfig)

# optimalcube


@dataclass
class OptimalCubeTrainConfig:
    dataset_artifact_name: str
    hard_budget: int  # not used, but pretend that it is
    seed: int
    output_artifact_aliases: list[str]


cs.store(name="train_optimalcube", node=OptimalCubeTrainConfig)


# zannone2019


@dataclass
class Zannone2019AgentConfig:
    # Value estimator parameters
    gamma: float
    lmbda: float

    # Loss parameters
    clip_epsilon: float
    entropy_bonus: bool
    entropy_coef: float
    critic_coef: float
    loss_critic_type: str

    # Optimization parameters
    num_epochs: int
    lr: float
    max_grad_norm: float
    replay_buffer_batch_size: int

    # Module parameters
    value_num_cells: list[int]
    value_dropout: float
    policy_num_cells: list[int]
    policy_dropout: float


@dataclass
class Zannone2019TrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    mdp: AFAMDPConfig
    rl_training_loop: AFARLTrainingLoopConfig
    soft_budget_param: float | None
    agent: Zannone2019AgentConfig
    n_generated_samples: (
        int  # how many artificial samples to generate using pretrained model
    )
    generation_batch_size: (
        int  # which batch size to use for artificial data generation
    )
    seed: int | None = None
    use_wandb: bool = False
    smoke_test: bool = False
    device: str | None = None

    # Aliases, only because snakefile assumes a flat interface
    hard_budget: int | None = None


cs.store(name="train_zannone2019", node=Zannone2019TrainConfig)

# kachuee2019


@dataclass
class Kachuee2019AgentConfig:
    # epsilon-greedy parameters
    eps_init: float
    eps_end: float
    eps_annealing_fraction: (
        float  # fraction of total number of expected batches
    )

    # How large batches should be sampled from replay buffer
    replay_buffer_batch_size: int
    replay_buffer_size: int  # how many samples fit in the replay buffer

    # Optimization parameters
    num_epochs: int  # how many times to pass over the batch of data received
    max_grad_norm: float
    lr: float
    update_tau: float

    # Loss parameters
    loss_function: str
    delay_value: bool
    double_dqn: bool

    # Value estimator parameters
    gamma: float
    lmbda: float


@dataclass
class Kachuee2019TrainConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    pretrained_model_bundle_path: str
    save_path: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig
    mdp: AFAMDPConfig
    rl_training_loop: AFARLTrainingLoopConfig
    soft_budget_param: float | None
    agent: Kachuee2019AgentConfig
    pretrained_model_lr: float
    activate_joint_training_after_fraction: float
    seed: int | None
    use_wandb: bool
    smoke_test: bool
    device: str | None
    # whether replay buffer device should be the same as `device`. If False, use cpu.
    replay_buffer_device_same_as_device: bool

    # Specific to kachuee2019
    reward_method: str  # one of {"softmax", "Bayesian-L1", "Bayesian-L2"}
    # how many samples to average over when calculating certainty for the reward
    mcdrop_samples: int

    # Aliases, only because snakefile assumes a flat interface
    hard_budget: int | None = None


cs.store(name="train_kachuee2019", node=Kachuee2019TrainConfig)
# ACO


@dataclass
class AACOConfig:
    k_neighbors: int = 5
    acquisition_cost: float = 0.05
    hide_val: float = 0.0  # Use 0 for consistency with MLP training
    evaluate_final_performance: bool = True
    eval_only_n_samples: int | None = None


@dataclass
class AACOTrainConfig:
    aco: AACOConfig
    dataset_artifact_name: Path
    save_path: Path
    classifier_bundle_path: Path | None = (
        None  # Path to pre-trained classifier bundle
    )
    seed: int = 42
    device: str = "cpu"
    cost_param: float | None = None
    hard_budget: int | None = None  # None = soft budget mode
    experiment_id: str | None = None
    initializer_type: str = "aaco"
    unmasker_type: str = "one_based_index"
    smoke_test: bool = False


@dataclass
class AACONNTrainConfig:
    """Config for AACO+NN (behavioral cloning) training."""

    aaco_bundle_path: Path  # Path to trained AACO method bundle
    dataset_artifact_name: Path  # Training dataset for rollout generation
    classifier_bundle_path: Path  # Path to pre-trained classifier bundle
    save_path: Path
    seed: int = 42
    device: str = "cpu"
    # Rollout generation
    max_acquisitions: int | None = (
        None  # Max acquisitions per sample during rollouts
    )
    # Policy network architecture
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    dropout: float = 0.1
    # Training parameters
    batch_size: int = 256
    max_epochs: int = 100
    learning_rate: float = 1e-3
    early_stopping_patience: int = 10
    val_split: float = 0.1  # Fraction of rollout data for validation
    hard_budget: int | None = None  # Hard budget for the trained policy
    smoke_test: bool = False


# --- TRAINING CLASSIFIERS ---


@dataclass
class TrainMaskedMLPClassifierConfig:
    train_dataset_path: str
    val_dataset_path: str
    save_path: str

    epochs: int
    batch_size: int
    limit_train_batches: int | None
    limit_val_batches: int | None
    min_masking_probability: float
    max_masking_probability: float

    initializer: InitializerConfig
    unmasker: UnmaskerConfig

    lr: float
    num_cells: list[int]
    dropout: float

    seed: int
    device: str
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(
    name="train_masked_mlp_classifier", node=TrainMaskedMLPClassifierConfig
)


@dataclass
class TrainMaskedViTClassifierConfig:
    train_dataset_bundle_path: str
    val_dataset_bundle_path: str
    save_path: str

    batch_size: int
    epochs: int
    min_masking_probability: float
    max_masking_probability: float
    # only_n_samples: int

    model_name: str
    image_size: int
    patch_size: int
    patience: int
    min_lr: float

    lr: float
    seed: int
    device: str
    initializer: InitializerConfig
    unmasker: UnmaskerConfig


cs.store(
    name="train_masked_vit_classifier", node=TrainMaskedViTClassifierConfig
)


# --- EVALUATION ---


@dataclass
class EvalConfig:
    # Which method to evaluate
    method_bundle_path: str
    # Which unmasker to use
    unmasker: UnmaskerConfig
    # Which initializer to use
    initializer: InitializerConfig
    # Which dataset instance to use
    dataset_bundle_path: str
    # Save path
    save_path: str
    # Also save results for predictions using an external classifier
    classifier_bundle_path: str | None
    seed: int | None
    device: str
    # Make it possible to only evaluate a subset of the dataset, for debugging purposes
    eval_only_n_samples: int | None
    batch_size: int
    # Set a hard budget during evaluation
    hard_budget: int | None = None
    # Whether to log to wandb
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="eval", node=EvalConfig)

# --- MISC ---


@dataclass
class TrainingTimeCalculationConfig:
    plotting_run_names: list[str]
    output_artifact_aliases: list[str]
    max_workers: int


cs.store(name="training_time_calculation", node=TrainingTimeCalculationConfig)


@dataclass
class EvaluationTimeCalculationConfig:
    plotting_run_names: list[str]
    output_artifact_aliases: list[str]
    max_workers: int


cs.store(
    name="evaluation_time_calculation", node=EvaluationTimeCalculationConfig
)


@dataclass
class PlotDownloadConfig:
    plotting_run_name: str
    datasets: list[str]  # only download plots of these datasets
    metrics: list[str]  # one metric per dataset
    budgets: list[
        str
    ]  # one list of budgets per dataset. A single '.' means that all budgets are accepted. Budgets are separated by whitespace.
    file_type: str  # e.g svg, png, pdf
    output_path: str  # where to store the downloaded figures


cs.store(name="training_time_calculation", node=TrainingTimeCalculationConfig)

# --- PLOTTING ---


@dataclass
class MetricConfig:
    key: str
    description: str
    ylim: list[int] | None


@dataclass
class PlotConfig:
    # path to a YAML config file which contains a list of evaluation artifacts to use
    eval_artifact_yaml_list: str
    metric_keys_and_descriptions: list[
        MetricConfig
    ]  # Inner list has two elements: [metric_key, description]
    max_workers: int  # how many parallel workers to use for loading evaluation artifacts


cs.store(name="plot", node=PlotConfig)
