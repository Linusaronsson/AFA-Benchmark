from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


@dataclass
class ResaveConfig:
    trained_model_bundle_path: str
    save_path: str

    device: str
    soft_budget_param: float


cs.store(name="resave", node=ResaveConfig)
