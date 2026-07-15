import importlib.util
from pathlib import Path
from types import ModuleType

from omegaconf import OmegaConf


def test_load_config_uses_pipeline_defaults() -> None:
    config_module = _load_workflow_config_module()

    loaded_config = config_module.load_config(_minimal_config())

    assert loaded_config["EVAL_DATASET_SPLIT"] == "test"
    assert loaded_config["DATASET_INSTANCE_INDICES"] == (0, 1, 2, 3, 4)
    assert loaded_config["SMOKE_TEST"] is False
    assert loaded_config["USE_WANDB"] is True
    assert loaded_config["DEVICE"] == "cpu"
    assert loaded_config["DEVICE_OVERRIDES"] == {}


def test_device_resolution_uses_most_specific_override() -> None:
    config_module = _load_workflow_config_module()
    overrides = {
        "datasets": {"cube": "mps"},
        "pretrained_models": {"pvae": "cuda:2"},
        "methods": {"aaco": "cuda:1"},
        "method_datasets": {"aaco": {"cube": "cuda:3"}},
    }

    assert (
        config_module.resolve_device(
            "cpu", overrides, dataset="cube", method="aaco"
        )
        == "cuda:3"
    )
    assert (
        config_module.resolve_device(
            "cpu", overrides, dataset="bank_marketing", method="aaco"
        )
        == "cuda:1"
    )
    assert (
        config_module.resolve_device(
            "cpu", overrides, dataset="cube", pretrained_model="pvae"
        )
        == "cuda:2"
    )
    assert (
        config_module.resolve_device("cpu", overrides, dataset="cube") == "mps"
    )
    assert (
        config_module.resolve_device("cpu", overrides, dataset="adult")
        == "cpu"
    )


def test_direct_unmasker_kwargs_are_empty_mapping() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "extra"
        / "conf"
        / "components"
        / "unmaskers"
        / "direct.yaml"
    )

    config = OmegaConf.load(config_path)

    assert dict(config.kwargs) == {}


def _load_workflow_config_module() -> ModuleType:
    config_path = (
        Path(__file__).parents[2] / "extra" / "workflow" / "src" / "config.py"
    )
    spec = importlib.util.spec_from_file_location(
        "workflow_config", config_path
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_config() -> dict[str, object]:
    return {
        "pretrain_mapping": {},
        "method_options": {},
        "methods": [],
        "datasets": ["cube"],
        "unmaskers": {"default": "direct"},
        "eval_hard_budgets": {"default": [1]},
        "soft_budget_params": {},
        "classifier_names": {"default": "masked_mlp_classifier"},
    }
