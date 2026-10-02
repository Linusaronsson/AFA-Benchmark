"""Workflow evaluation batch sizes: pinned for AACO, loud when missing."""

import importlib.util
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parents[2]


def test_aaco_eval_batch_size_is_pinned_per_dataset() -> None:
    """AACO batches its acquisition, so all.yaml pins its eval batch size."""
    config_module = _load_workflow_config_module()
    datasets = ["cube", "mnist", "fashion_mnist", "synthetic_mnist"]
    config = _config_with_method_options(
        _all_method_options(), methods=["aaco", "aaco_nn"], datasets=datasets
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded_config = config_module.load_config(config)

    # Image datasets get the smaller pinned size, every other dataset the
    # default. The loader keeps the `default` key alongside the datasets.
    expected = {
        "default": 128,
        "cube": 128,
        "mnist": 32,
        "fashion_mnist": 32,
        "synthetic_mnist": 32,
        "synthetic_mnist_without_noise": 32,
    }
    assert loaded_config["EVAL_BATCH_SIZES"]["aaco"] == expected
    assert loaded_config["EVAL_BATCH_SIZES"]["aaco_nn"] == expected


def test_missing_eval_batch_size_warns_and_falls_back_to_one() -> None:
    config_module = _load_workflow_config_module()
    config = _config_with_method_options(
        {"unbatched": {"train_script_name": "unbatched"}},
        methods=["unbatched"],
        datasets=["cube", "mnist"],
    )

    with pytest.warns(UserWarning, match="unbatched.*eval_batch_size"):
        loaded_config = config_module.load_config(config)

    assert loaded_config["EVAL_BATCH_SIZES"]["unbatched"] == {
        "cube": 1,
        "mnist": 1,
    }


def _all_method_options() -> dict[str, Any]:
    config_path = (
        REPO_ROOT
        / "extra"
        / "workflow"
        / "conf"
        / "method_options"
        / "all.yaml"
    )
    container = OmegaConf.to_container(OmegaConf.load(config_path))
    assert isinstance(container, dict)
    return container["method_options"]


def _load_workflow_config_module() -> ModuleType:
    config_path = REPO_ROOT / "extra" / "workflow" / "src" / "config.py"
    spec = importlib.util.spec_from_file_location(
        "workflow_config", config_path
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_with_method_options(
    method_options: dict[str, Any],
    *,
    methods: list[str],
    datasets: list[str],
) -> dict[str, object]:
    return {
        "pretrain_mapping": {"aaco": {"pretrain_script_name": "aaco"}},
        "method_options": method_options,
        "methods": methods,
        "datasets": datasets,
        "unmaskers": {"default": "direct"},
        "eval_hard_budgets": {"default": [1]},
        "soft_budget_params": {
            method: {"default": [[0.1, 0.1]]} for method in methods
        },
        "classifier_names": {"default": "masked_mlp_classifier"},
    }
