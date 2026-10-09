from dataclasses import dataclass

from afabench.fit.contract import PretrainingContract, store_contract_config


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
    # "real" or "synthetic", see `sample_task`. A plain string because
    # OmegaConf structured configs reject `Literal` fields.
    feature_source: str
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


store_contract_config(name="pretrain_l2m", config_class=L2MPretrainingConfig)
