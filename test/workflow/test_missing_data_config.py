import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    path = (
        Path(__file__).parents[2]
        / "extra"
        / "workflow"
        / "src"
        / "missing_data_config.py"
    )
    spec = importlib.util.spec_from_file_location("missing_data_config", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_method_variants_inherit_shared_method_configuration() -> None:
    module = _load_module()
    options = {
        "jafa": {
            "train_script_name": "jafa",
            "pretrained_model_name": "pvae",
            "method_specific_params": ["model.hidden_dim=64"],
        },
        "odin_model_free": {
            "train_script_name": "odin_model_free",
            "pretrained_model_name": "odin_model_free",
        },
    }
    specs = module.build_method_specs(
        ["jafa"],
        options,
        {"jafa": {"extra_strategies": ["zero_fill"]}},
        {
            "jafa_full_state": {
                "base_method": "jafa",
                "allowed_strategies": ["restricted"],
                "train_params": ["agent.use_action_availability=true"],
            },
            "odin_model_free_full_state": {
                "base_method": "odin_model_free",
            },
        },
    )

    assert set(specs) == {"jafa", "jafa_full_state"}
    assert specs["jafa"].extra_strategies == ("zero_fill",)
    control = specs["jafa_full_state"]
    assert control.base_method == "jafa"
    assert control.train_script_name == "jafa"
    assert control.pretrained_model_name == "pvae"
    assert control.allowed_strategies == ("restricted",)
    assert control.train_params == (
        "model.hidden_dim=64",
        "agent.use_action_availability=true",
    )
    assert control.include_complete_data is False


def test_largest_hard_budget_honors_train_eval_mapping() -> None:
    module = _load_module()

    selected = module.largest_hard_budget(
        [
            (4, 4, "null", "null"),
            (20, 14, "null", "null"),
            ("null", "null", 0.1, 0.1),
        ]
    )

    assert selected == ("20", "14")


def test_true_completion_reuses_the_seeded_complete_policy() -> None:
    module = _load_module()

    assert module.policy_training_coordinates(
        "mnar_self", "0.7", "true_completion"
    ) == ("none", "0.0", "complete")
    assert module.policy_training_coordinates(
        "mnar_self", "0.7", "pvae_oracle"
    ) == ("mnar_self", "0.7", "pvae_oracle")


def test_strategy_filter_selects_only_declared_cells() -> None:
    module = _load_module()
    filters = {
        "pvae_stepwise": {
            "datasets": ["cube", "cube_nm"],
            "methods": ["jafa", "odin_model_free", "ol_without_mask"],
            "mechanisms": ["mcar"],
            "probabilities": [0.3, 0.5, 0.7],
        }
    }

    assert module.strategy_enabled(
        filters,
        "pvae_stepwise",
        dataset="cube_nm",
        method="jafa",
        base_method="jafa",
        mechanism="mcar",
        probability="0.5",
    )
    assert not module.strategy_enabled(
        filters,
        "pvae_stepwise",
        dataset="actg",
        method="jafa",
        base_method="jafa",
        mechanism="mcar",
        probability="0.5",
    )
    assert not module.strategy_enabled(
        filters,
        "pvae_stepwise",
        dataset="cube_nm",
        method="jafa",
        base_method="jafa",
        mechanism="mar",
        probability="0.5",
    )


def test_strategy_filter_accepts_base_method_for_variants() -> None:
    module = _load_module()

    assert module.strategy_enabled(
        {"control": {"methods": ["jafa"]}},
        "control",
        dataset="cube",
        method="aaco_variant",
        base_method="jafa",
        mechanism="mcar",
        probability="0.3",
    )
