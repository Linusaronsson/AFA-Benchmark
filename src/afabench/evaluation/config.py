from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig

cs = ConfigStore.instance()


@dataclass
class EvalConfig:
    method_bundle_path: str
    unmasker: UnmaskerConfig
    initializer: InitializerConfig
    dataset_bundle_path: str
    save_path: str
    classifier_bundle_path: str | None
    seed: int | None
    device: str
    eval_only_n_samples: int | None
    batch_size: int
    hard_budget: int | None = None
    soft_budget_param: float | None = None
    use_wandb: bool = False
    smoke_test: bool = False


cs.store(name="eval", node=EvalConfig)
