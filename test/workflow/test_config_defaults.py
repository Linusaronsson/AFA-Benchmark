import importlib.util
from pathlib import Path
from types import ModuleType

from omegaconf import OmegaConf

from afabench.plotting.methods import PRIMARY_METHODS
from scripts.workflow.write_run_manifest import resolved_config


def test_current_study_matches_the_reported_method_matrix() -> None:
    config = resolved_config(Path(__file__).parents[2], "missing_data", [])
    methods = set(config["methods"])
    methods.update(
        name
        for name, variant in config["missing_data_method_variants"].items()
        if variant["base_method"] in methods
    )
    assert methods == set(PRIMARY_METHODS)
    assert len(config["datasets"]) == 8
    assert config["dataset_instance_indices"] == list(range(5))
    assert config["strategies"] == ["restricted", "pvae_label_conditioned"]
    assert config["missingness"]["probabilities"] == [0.3, 0.5, 0.7]
    assert len(config["missingness"]["mechanisms"]) == 4
    assert config["artifact_namespace"] == "induced"
    assert config["eval_dataset_split"] == "val"
    assert config["paper_artifacts"]


def test_non_smoke_missing_data_configs_keep_canonical_cube_size() -> None:
    root = Path(__file__).parents[2]
    config_dir = root / "extra" / "workflow" / "conf" / "missing_data"
    cube_datasets = {"cube", "cube_nm", "cube_nonuniform_costs"}

    for path in config_dir.glob("*.yaml"):
        if "smoke" in path.stem:
            continue
        config = OmegaConf.load(path)
        generation = config.get("dataset_generation_params", {})
        for dataset in cube_datasets:
            params = generation.get(dataset, [])
            assert not any(
                str(param).startswith("dataset.kwargs.n_samples=")
                for param in params
            ), path


def test_missing_data_ol_variants_share_calibrated_training_budget() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "extra"
        / "workflow"
        / "conf"
        / "missing_data"
        / "design.yaml"
    )
    config = OmegaConf.load(config_path)
    runtime_params = config.train_runtime_params
    expected = [
        "rl_training_loop.n_batches=2000",
        "rl_training_loop.frames_per_batch=256",
        "rl_training_loop.eval_n_times=10",
    ]

    for method in ("ol_with_mask", "ol_full_state"):
        assert list(runtime_params[method]) == expected


def test_stepwise_evaluation_uses_memory_safe_batches() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "extra"
        / "workflow"
        / "conf"
        / "missing_data"
        / "design.yaml"
    )
    config = OmegaConf.load(config_path)

    assert config.stepwise_eval_batch_size == 16


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
