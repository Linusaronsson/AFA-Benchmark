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
        "aaco": {
            "train_script_name": "aaco",
            "pretrained_model_name": "pvae",
            "method_specific_params": ["model.hidden_dim=64"],
        },
        "dime": {
            "train_script_name": "dime",
            "pretrained_model_name": "dime",
        },
    }
    specs = module.build_method_specs(
        ["aaco"],
        options,
        {"aaco": {"extra_strategies": ["zero_fill"]}},
        {
            "aaco_doubly_robust": {
                "base_method": "aaco",
                "allowed_strategies": ["restricted"],
                "train_params": ["method.doubly_robust=true"],
            },
            "dime_feature_marginal_ipw": {
                "base_method": "dime",
            },
        },
    )

    assert set(specs) == {"aaco", "aaco_doubly_robust"}
    assert specs["aaco"].extra_strategies == ("zero_fill",)
    control = specs["aaco_doubly_robust"]
    assert control.base_method == "aaco"
    assert control.train_script_name == "aaco"
    assert control.pretrained_model_name == "pvae"
    assert control.allowed_strategies == ("restricted",)
    assert control.train_params == (
        "model.hidden_dim=64",
        "method.doubly_robust=true",
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


def test_strategy_filter_selects_only_declared_cells() -> None:
    module = _load_module()
    filters = {
        "pvae_stepwise": {
            "datasets": ["cube", "cube_nm"],
            "methods": ["aaco", "dime", "ol_without_mask"],
            "mechanisms": ["mcar"],
            "probabilities": [0.3, 0.5, 0.7],
        }
    }

    assert module.strategy_enabled(
        filters,
        "pvae_stepwise",
        dataset="cube_nm",
        method="aaco",
        base_method="aaco",
        mechanism="mcar",
        probability="0.5",
    )
    assert not module.strategy_enabled(
        filters,
        "pvae_stepwise",
        dataset="actg",
        method="aaco",
        base_method="aaco",
        mechanism="mcar",
        probability="0.5",
    )
    assert not module.strategy_enabled(
        filters,
        "pvae_stepwise",
        dataset="cube_nm",
        method="aaco",
        base_method="aaco",
        mechanism="mar",
        probability="0.5",
    )


def test_strategy_filter_accepts_base_method_for_variants() -> None:
    module = _load_module()

    assert module.strategy_enabled(
        {"control": {"methods": ["aaco"]}},
        "control",
        dataset="cube",
        method="aaco_variant",
        base_method="aaco",
        mechanism="mcar",
        probability="0.3",
    )
