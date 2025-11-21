from enum import Enum
from typing import Any
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()

# --- DATASETS ---


@dataclass
class SplitRatioConfig:
    train: float  # ratio of training data
    val: float  # ratio of validation data
    test: float  # ratio of test data


class DatasetType(str, Enum):
    cube = "cube"
    MNIST = "MNIST"
    shim2018cube = "shim2018cube"
    diabetes = "diabetes"
    physionet = "physionet"
    miniboone = "miniboone"
    FashionMNIST = "FashionMNIST"
    bank_marketing = "bank_marketing"
    ckd = "ckd"
    actg = "actg"


@dataclass
class DatasetConfig:
    type: DatasetType
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetGenerationConfig:
    data_dir: str
    split_idx: list[int]
    seeds: list[int]
    split_ratio: SplitRatioConfig
    # Small float added to standard deviation to avoid division by zero
    epsilon: float
    dataset: DatasetConfig


cs.store(name="dataset_generation", node=DatasetGenerationConfig)

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
    dataset_artifact_name: str
    batch_size: int  # batch size for dataloader
    epochs: int
    limit_train_batches: int | None
    limit_val_batches: int | None

    device: str
    seed: int
    lr: float
    min_masking_probability: float
    max_masking_probability: float
    encoder: Shim2018EncoderConfig
    classifier: Shim2018ClassifierConfig


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
    dataset_artifact_name: str
    batch_size: int  # batch size for dataloader
    epochs: int
    limit_train_batches: int | None
    limit_val_batches: int | None

    device: str
    seed: int
    lr: float
    min_masking_probability: float
    max_masking_probability: float
    pointnet: Zannone2019PointNetConfig
    encoder: Zannone2019EncoderConfig
    partial_vae: Zannone2019PartialVAEConfig
    classifier: Zannone2019ClassifierConfig
    start_kl_scaling_factor: float
    end_kl_scaling_factor: float
    n_annealing_epochs: int
    classifier_loss_scaling_factor: float


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
    dataset_artifact_name: str
    batch_size: int  # batch size for dataloader
    epochs: int
    limit_train_batches: int | None
    limit_val_batches: int | None

    device: str
    seed: int
    lr: float
    min_masking_probability: float
    max_masking_probability: float
    pq_module: Kachuee2019PQModuleConfig


cs.store(name="pretrain_kachuee2019", node=Kachuee2019PretrainConfig)


# --- TRAINING METHODS ---

# shim2018


@dataclass
class Shim2018AgentConfig:
    # epsilon-greedy parameters
    eps_init: float
    eps_end: float
    eps_annealing_num_batches: int

    # How large batches should be sampled from replay buffer
    replay_buffer_batch_size: int

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
    pretrained_model_artifact_name: str
    n_agents: int
    hard_budget: int | None
    cost_param: float | None  # how much each feature costs
    agent: Shim2018AgentConfig
    n_batches: int  # how many batches to train the agent
    batch_size: int  # batch size for collector
    eval_every_n_batches: int | None  # how often to evaluate the agent
    eval_max_steps: (
        int  # maximum allowed number of steps in an evaluation episode
    )
    n_eval_episodes: int  # how many episodes to average over in evaluation

    device: str
    seed: int | None
    pretrained_model_lr: float
    activate_joint_training_after_n_batches: int
    output_artifact_aliases: list[str]
    evaluate_final_performance: bool
    eval_only_n_samples: int | None


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
    pretrained_model_artifact_name: str
    hard_budget: int
    device: str = "cuda"
    seed: int = 42
    output_artifact_aliases: list[str] = field(default_factory=list)


cs.store(name="train_ma2018", node=Ma2018TrainingConfig)


@dataclass
class Covert2023PretrainingConfig:
    dataset_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    seed: int = 42
    device: str = "cuda"
    lr: float = 1e-3
    nepochs: int = 100
    patience: int = 5
    min_masking_probability: float = 0.0
    max_masking_probability: float = 0.9

    hidden_units: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.3
    activations: str = "ReLU"
    flag_drop_out: bool = True
    flag_only_output_layer: bool = False
    experiment_id: str | None = None


cs.store(name="pretrain_covert2023", node=Covert2023PretrainingConfig)


@dataclass
class Covert2023Pretraining2DConfig:
    dataset_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    seed: int = 42
    device: str = "cuda"
    lr: float = 1e-5
    nepochs: int = 50
    patience: int = 2
    min_masking_probability: float = 0.0
    max_masking_probability: float = 0.9

    image_size: int = 224
    patch_size: int = 16


cs.store(name="pretrain_covert2023", node=Covert2023Pretraining2DConfig)


@dataclass
class Covert2023TrainingConfig:
    pretrained_model_artifact_name: str
    batch_size: int = 128
    lr: float = 1e-3
    hard_budget: int = 20
    nepochs: int = 100
    patience: int = 5
    device: str = "cuda"
    seed: int = 42
    experiment_id: str | None = None

    hidden_units: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.3
    activations: str = "ReLU"
    flag_drop_out: bool = True
    flag_only_output_layer: bool = False


cs.store(name="train_covert2023", node=Covert2023TrainingConfig)


@dataclass
class Covert2023Training2DConfig:
    pretrained_model_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    lr: float = 1e-5
    min_lr: float = 1e-8
    hard_budget: int = 20
    nepochs: int = 50
    patience: int = 3
    device: str = "cuda"
    seed: int = 42


cs.store(name="train_covert2023", node=Covert2023Training2DConfig)


@dataclass
class Gadgil2023PretrainingConfig:
    dataset_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    seed: int = 42
    device: str = "cuda"
    lr: float = 1e-3
    nepochs: int = 100
    patience: int = 5
    min_masking_probability: float = 0.0
    max_masking_probability: float = 0.9

    hidden_units: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.3
    activations: str = "ReLU"
    flag_drop_out: bool = True
    flag_only_output_layer: bool = False


cs.store(name="pretrain_gadgil2023", node=Gadgil2023PretrainingConfig)


@dataclass
class Gadgil2023Pretraining2DConfig:
    dataset_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    seed: int = 42
    device: str = "cuda"
    lr: float = 1e-5
    nepochs: int = 50
    patience: int = 2
    min_masking_probability: float = 0.0
    max_masking_probability: float = 0.9

    image_size: int = 224
    patch_size: int = 16


cs.store(name="pretrain_gadgil2023", node=Gadgil2023Pretraining2DConfig)


@dataclass
class Gadgil2023TrainingConfig:
    pretrained_model_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    lr: float = 1e-3
    hard_budget: int = 20
    nepochs: int = 100
    patience: int = 5
    eps: float = 0.05
    eps_decay: float = 0.2
    eps_steps: int = 10
    device: str = "cuda"
    seed: int = 42

    hidden_units: list[int] = field(default_factory=lambda: [128, 128])
    dropout: float = 0.3
    activations: str = "ReLU"
    flag_drop_out: bool = True
    flag_only_output_layer: bool = False


cs.store(name="train_gadgil2023", node=Gadgil2023TrainingConfig)


@dataclass
class Gadgil2023Training2DConfig:
    pretrained_model_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    lr: float = 1e-5
    min_lr: float = 1e-8
    hard_budget: int = 20
    nepochs: int = 250
    patience: int = 3
    eps: float = 0.05
    eps_decay: float = 0.2
    eps_steps: int = 10
    device: str = "cuda"
    seed: int = 42


cs.store(name="train_gadgil2023", node=Gadgil2023Training2DConfig)


@dataclass
class StaticSelectorConfig:
    lr: float = 1e-3
    nepochs: int = 250
    num_cells: list[int] = field(default_factory=lambda: [128, 128])
    patience: int = 5


@dataclass
class StaticClassifierConfig:
    lr: float = 1e-3
    nepochs: int = 250
    num_cells: list[int] = field(default_factory=lambda: [128, 128])


@dataclass
class CAETrainingConfig:
    dataset_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    hard_budget: int = 20
    device: str = "cuda"
    seed: int = 42

    selector: StaticSelectorConfig = field(
        default_factory=StaticSelectorConfig
    )
    classifier: StaticClassifierConfig = field(
        default_factory=StaticClassifierConfig
    )


cs.store(name="train_cae", node=CAETrainingConfig)


@dataclass
class PermutationTrainingConfig:
    dataset_artifact_name: str
    output_artifact_aliases: list[str] = field(default_factory=list)

    batch_size: int = 128
    hard_budget: int = 20
    device: str = "cuda"
    seed: int = 42

    selector: StaticSelectorConfig = field(
        default_factory=StaticSelectorConfig
    )
    classifier: StaticClassifierConfig = field(
        default_factory=StaticClassifierConfig
    )


cs.store(name="train_permutation", node=PermutationTrainingConfig)

# randomdummy


@dataclass
class RandomDummyTrainConfig:
    dataset_artifact_name: str
    hard_budget: int | None  # not used, but pretend that it is
    device: str
    seed: int
    output_artifact_aliases: list[str]
    cost_param: float
    experiment_id: str | None = None


cs.store(name="train_randomdummy", node=RandomDummyTrainConfig)

# sequentialdummy


@dataclass
class SequentialDummyTrainConfig:
    dataset_artifact_name: str
    hard_budget: int | None  # not used, but pretend that it is
    device: str
    seed: int | None
    output_artifact_aliases: list[str]
    cost_param: float


cs.store(name="train_randomdummy", node=RandomDummyTrainConfig)

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
    pretrained_model_artifact_name: str
    n_agents: int
    hard_budget: int | None
    cost_param: float | None
    agent: Zannone2019AgentConfig
    n_batches: int  # how many batches to train the agent
    batch_size: int  # batch size for collector
    eval_every_n_batches: int | None  # how often to evaluate the agent
    eval_max_steps: (
        int  # maximum allowed number of steps in an evaluation episode
    )
    n_eval_episodes: int  # how many episodes to average over in evaluation
    n_generated_samples: (
        int  # how many artificial samples to generate using pretrained model
    )
    generation_batch_size: (
        int  # which batch size to use for artificial data generation
    )

    device: str
    seed: int
    output_artifact_aliases: list[str]
    evaluate_final_performance: bool
    eval_only_n_samples: int | None
    visualize: bool


cs.store(name="train_zannone2019", node=Zannone2019TrainConfig)

# kachuee2019


@dataclass
class Kachuee2019AgentConfig:
    # epsilon-greedy parameters
    eps_init: float
    eps_end: float
    eps_annealing_num_batches: int

    # How large batches should be sampled from replay buffer
    replay_buffer_batch_size: int
    replay_buffer_size: int  # how many samples fit in the replay buffer

    # Optimization parameters
    num_optim: int  # how many batches to sample from replay buffer
    max_action_value_grad_norm: float
    action_value_lr: float
    update_tau: float
    max_classification_grad_norm: float
    classification_lr: float  # with cross entropy loss

    # Loss parameters
    loss_function: str
    delay_value: bool
    double_dqn: bool

    # Value estimator parameters
    gamma: float


@dataclass
class Kachuee2019TrainConfig:
    reward_method: str  # one of {"softmax", "Bayesian-L1", "Bayesian-L2"}
    # how many samples to average over when calculating certainty for the reward
    mcdrop_samples: int

    pretrained_model_artifact_name: str
    n_agents: int
    hard_budget: int | None
    cost_param: float | None
    agent: Kachuee2019AgentConfig
    n_batches: int  # how many batches to train the agent
    batch_size: int  # batch size for collector
    eval_every_n_batches: int | None  # how often to evaluate the agent
    eval_max_steps: (
        int  # maximum allowed number of steps in an evaluation episode
    )
    n_eval_episodes: int  # how many episodes to average over in evaluation

    device: str
    seed: int
    output_artifact_aliases: list[str]
    evaluate_final_performance: bool
    eval_only_n_samples: int | None


cs.store(name="train_kachuee2019", node=Kachuee2019TrainConfig)

# ACO


@dataclass
class AACOConfig:
    k_neighbors: int = 5
    acquisition_cost: float = 0.05
    hide_val: float = 10.0
    evaluate_final_performance: bool = True
    eval_only_n_samples: int | None = None


@dataclass
class AACOTrainConfig:
    aco: AACOConfig
    dataset_artifact_name: str
    seed: int = 42
    device: str = "cpu"
    cost_param: float | None = None
    hard_budget: float | None = None
    experiment_id: str | None = None


# --- TRAINING CLASSIFIERS ---


@dataclass
class TrainMaskedMLPClassifierConfig:
    dataset_artifact_name: str
    batch_size: int
    epochs: int
    limit_train_batches: int | None
    limit_val_batches: int | None
    min_masking_probability: float
    max_masking_probability: float
    eval_only_n_samples: (
        int | None
    )  # if specified, only evaluate on this many samples
    lr: float
    seed: int
    device: str
    num_cells: list[int]
    dropout: float
    output_artifact_aliases: list[str]
    evaluate_final_performance: bool


cs.store(
    name="train_masked_mlp_classifier", node=TrainMaskedMLPClassifierConfig
)


@dataclass
class TrainMaskedViTClassifierConfig:
    dataset_artifact_name: str
    batch_size: int
    epochs: int
    min_masking_probability: float
    max_masking_probability: float

    model_name: str
    image_size: int
    patch_size: int
    patience: int
    min_lr: float

    lr: float
    seed: int
    device: str
    output_artifact_aliases: list[str]


cs.store(
    name="train_masked_vit_classifier", node=TrainMaskedViTClassifierConfig
)


# --- EVALUATION ---


@dataclass
class EvalConfig:
    trained_method_artifact_name: str
    # if None, use the method's classifier
    trained_classifier_artifact_name: str | None
    seed: int
    device: str
    output_artifact_aliases: list[str]
    eval_only_n_samples: (
        int | None
    )  # if specified, only evaluate on this many samples
    batch_size: int
    dataset_split: str  # use "validation" or "testing"
    budget: int | None = (
        None  # if specified, override the budget from training
    )


cs.store(name="eval", node=EvalConfig)


@dataclass
class SoftEvalConfig:
    trained_method_artifact_name: str
    cost_param: (
        float | None
    )  # Some AFAMethods don't need a cost parameter during training, but need it during evaluation
    # if None, use the method's classifier
    trained_classifier_artifact_name: str  # has to be given
    seed: int | None
    device: str
    eval_only_n_samples: (
        int | None
    )  # if specified, only evaluate on this many samples
    dataset_split: str  # use "validation" or "testing"
    batch_size: int
    budget: int | None = (
        None  # if specified, override the budget from training
    )
    experiment_id: str | None = None


cs.store(name="soft-eval", node=SoftEvalConfig)

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
