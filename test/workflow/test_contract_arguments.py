"""
The Snakemake renderer turns the training contract into command-line arguments.

`extra/workflow/src/contract_arguments.py` reads the field names from the
contract dataclasses in `afabench.fit.contract`, which the Snakefile
imports at parse time.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).parents[2]

PRETRAINING_VALUES = {
    "train_dataset_bundle_path": "train.bundle",
    "val_dataset_bundle_path": "val.bundle",
    "classifier_bundle_path": "classifier.bundle",
    "save_path": "model.bundle",
    "initializer": "cold",
    "unmasker": "direct",
    "dataset_key": "cube",
    "device": "cpu",
    "seed": 3,
    "use_wandb": False,
    "smoke_test": True,
}

TRAINING_VALUES = {
    **PRETRAINING_VALUES,
    "save_path": "method.bundle",
    "method_name": "my_method",
    "pretrained_model_bundle_path": "model.bundle",
    "hard_budget": 5,
    "soft_budget_param": "null",
}


def _load_contract_arguments_module() -> ModuleType:
    module_path = REPO_ROOT / "extra/workflow/src/contract_arguments.py"
    spec = importlib.util.spec_from_file_location(
        "workflow_contract_arguments", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_arguments(rendered: str) -> dict[str, str]:
    arguments = rendered.split()
    parsed = dict(argument.split("=", maxsplit=1) for argument in arguments)
    assert len(parsed) == len(arguments), "each key is rendered once"
    return parsed


@pytest.fixture(scope="module")
def renderer() -> ModuleType:
    return _load_contract_arguments_module()


def test_contract_import_does_not_load_torch_or_sklearn() -> None:
    # Snakemake imports the contract and the output layout on every parse,
    # including dry runs.
    probe = (
        "import sys\n"
        "import afabench.fit.contract\n"
        "import afabench.core.output_layout\n"
        "print(sorted({'torch', 'sklearn'} & set(sys.modules)))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.strip() == "[]"


def test_training_contract_renders_one_argument_per_field(
    renderer: ModuleType,
) -> None:
    rendered = renderer.render_training_contract(TRAINING_VALUES)

    assert _parse_arguments(rendered) == {
        name: str(value) for name, value in TRAINING_VALUES.items()
    }


def test_pretraining_contract_renders_one_argument_per_field(
    renderer: ModuleType,
) -> None:
    rendered = renderer.render_pretraining_contract(PRETRAINING_VALUES)

    assert _parse_arguments(rendered) == {
        name: str(value) for name, value in PRETRAINING_VALUES.items()
    }


def test_training_contract_omits_pretrained_model_without_pretraining_stage(
    renderer: ModuleType,
) -> None:
    values = {**TRAINING_VALUES, "pretrained_model_bundle_path": None}

    rendered = renderer.render_training_contract(values)

    assert "pretrained_model_bundle_path" not in _parse_arguments(rendered)


def test_training_contract_rejects_a_missing_field(
    renderer: ModuleType,
) -> None:
    values = {
        name: value
        for name, value in TRAINING_VALUES.items()
        if name != "hard_budget"
    }

    with pytest.raises(ValueError, match="hard_budget"):
        renderer.render_training_contract(values)


def test_pretraining_contract_rejects_training_only_fields(
    renderer: ModuleType,
) -> None:
    with pytest.raises(ValueError, match="hard_budget"):
        renderer.render_pretraining_contract(TRAINING_VALUES)


def test_every_workflow_dataset_has_a_dataset_key_config() -> None:
    dataset_lists = (REPO_ROOT / "extra/workflow/conf/datasets").glob("*.yaml")
    dataset_keys = {
        dataset_key
        for dataset_list in dataset_lists
        for dataset_key in OmegaConf.load(dataset_list)["datasets"]
    }

    missing = {
        dataset_key
        for dataset_key in dataset_keys
        if not (
            REPO_ROOT
            / "extra/conf/components/dataset_key"
            / f"{dataset_key}.yaml"
        ).is_file()
    }

    assert missing == set()
