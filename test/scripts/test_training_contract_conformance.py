"""
Every pretraining and training script accepts the training contract.

Each method name in the workflow's `method_options` is run as a subprocess
with exactly the arguments the Snakemake renderer produces, on a generated
smoke CUBE dataset, and must write a loadable bundle to `save_path`. All
cases are marked `pipeline` except `random_dummy`, which keeps the default
suite exercising the contract end to end.
"""

import importlib.util
import inspect
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType

import pytest
import torch
from _pytest.mark import ParameterSet
from omegaconf import OmegaConf

from afabench.components.classifiers import WrappedMaskedMLPClassifier
from afabench.components.classifiers.models import MaskedMLPClassifier
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.core.registry import get_class
from afabench.datasets.datasets import CubeDataset

REPO_ROOT = Path(__file__).parents[2]
WORKFLOW_CONF = REPO_ROOT / "extra/workflow/conf"
METHOD_OPTIONS: Mapping[str, Mapping[str, object]] = OmegaConf.to_container(  # pyright: ignore[reportAssignmentType]
    OmegaConf.load(WORKFLOW_CONF / "method_options/all.yaml")["method_options"]
)
PRETRAIN_MAPPING: Mapping[str, Mapping[str, object]] = OmegaConf.to_container(  # pyright: ignore[reportAssignmentType]
    OmegaConf.load(WORKFLOW_CONF / "pretrain_mappings/all.yaml")[
        "pretrain_mapping"
    ]
)
DEFAULT_SUITE_METHODS = {"random_dummy"}

DATASET_KEY = "cube"
HARD_BUDGET = 3
SEED = 0
CPU = torch.device("cpu")
SCRIPT_TIMEOUT_SECONDS = 1800


def _method_cases() -> list[ParameterSet]:
    return [
        pytest.param(
            method_name,
            marks=[]
            if method_name in DEFAULT_SUITE_METHODS
            else [pytest.mark.pipeline],
            id=method_name,
        )
        for method_name in METHOD_OPTIONS
    ]


def _load_renderer() -> ModuleType:
    module_path = REPO_ROOT / "extra/workflow/src/contract_arguments.py"
    spec = importlib.util.spec_from_file_location(
        "workflow_contract_arguments", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_script(
    script: str, contract_arguments: str, extra_arguments: str, run_dir: Path
) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / script),
        *contract_arguments.split(),
        *extra_arguments.split(),
        f"hydra.run.dir={run_dir}",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=SCRIPT_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, (
        f"{' '.join(command)}\n--- stdout ---\n{result.stdout[-4000:]}"
        f"\n--- stderr ---\n{result.stderr[-4000:]}"
    )


def _load_bundle_on_cpu(bundle_path: Path) -> object:
    """
    Load a bundle the way its consumer would, on the CPU.

    Bundle classes name the target device differently (`device`,
    `map_location` or `**kwargs`), so the keyword is read off the class.
    """
    manifest = json.loads((bundle_path / "manifest.json").read_text())
    load_parameters = inspect.signature(
        get_class(manifest["class_name"]).load
    ).parameters
    accepts_any_keyword = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in load_parameters.values()
    )
    device_keywords = {
        name: CPU
        for name in ("device", "map_location")
        if name in load_parameters
        or (accepts_any_keyword and name == "device")
    }
    loaded, _ = load_bundle(bundle_path, **device_keywords)
    return loaded


class SmokeInputs:
    """Generated CUBE bundles shared by every case in the session."""

    def __init__(self, root: Path) -> None:
        self.root: Path = root
        self.train_dataset_bundle_path: Path = root / "train.bundle"
        self.val_dataset_bundle_path: Path = root / "val.bundle"
        self.classifier_bundle_path: Path = root / "classifier.bundle"
        self.pretrained_model_bundle_paths: dict[str, Path] = {}

        # Large enough for the methods that keep their own batch size of 128
        # with drop_last during a smoke test.
        train_dataset = CubeDataset(n_samples=256, seed=SEED)
        save_bundle(train_dataset, self.train_dataset_bundle_path, metadata={})
        save_bundle(
            CubeDataset(n_samples=64, seed=SEED + 1),
            self.val_dataset_bundle_path,
            metadata={},
        )
        save_bundle(
            WrappedMaskedMLPClassifier(
                MaskedMLPClassifier(
                    n_features=train_dataset.feature_shape.numel(),
                    n_classes=train_dataset.label_shape.numel(),
                    num_cells=(8,),
                ),
                device=CPU,
            ),
            self.classifier_bundle_path,
            metadata={},
        )

    def pretraining_values(self, save_path: Path) -> dict[str, object]:
        return {
            "train_dataset_bundle_path": self.train_dataset_bundle_path,
            "val_dataset_bundle_path": self.val_dataset_bundle_path,
            "classifier_bundle_path": self.classifier_bundle_path,
            "save_path": save_path,
            "initializer": "cold",
            "unmasker": "direct",
            "dataset_key": DATASET_KEY,
            "device": "cpu",
            "seed": SEED,
            "use_wandb": False,
            "smoke_test": True,
        }

    def training_values(
        self, save_path: Path, pretrained_model_bundle_path: Path | None
    ) -> dict[str, object]:
        return {
            **self.pretraining_values(save_path),
            "pretrained_model_bundle_path": pretrained_model_bundle_path,
            "hard_budget": HARD_BUDGET,
            "soft_budget_param": "null",
        }


@pytest.fixture(scope="session")
def smoke_inputs(tmp_path_factory: pytest.TempPathFactory) -> SmokeInputs:
    return SmokeInputs(tmp_path_factory.mktemp("training_contract"))


def _pretrained_model_bundle_path(
    smoke_inputs: SmokeInputs, pretrained_model_name: str
) -> Path:
    """Pretrain once per pretrained model name, as the pipeline shares them."""
    if pretrained_model_name not in smoke_inputs.pretrained_model_bundle_paths:
        renderer = _load_renderer()
        mapping = PRETRAIN_MAPPING[pretrained_model_name]
        save_path = smoke_inputs.root / pretrained_model_name / "model.bundle"
        _run_script(
            f"pretrain_model/{mapping['pretrain_script_name']}.py",
            renderer.render_pretraining_contract(
                smoke_inputs.pretraining_values(save_path)
            ),
            " ".join(mapping.get("pretrain_params", [])),  # pyright: ignore[reportArgumentType, reportCallIssue]
            smoke_inputs.root / pretrained_model_name / "hydra",
        )
        assert _load_bundle_on_cpu(save_path) is not None
        smoke_inputs.pretrained_model_bundle_paths[pretrained_model_name] = (
            save_path
        )
    return smoke_inputs.pretrained_model_bundle_paths[pretrained_model_name]


@pytest.mark.parametrize("method_name", _method_cases())
def test_training_script_writes_a_loadable_bundle_at_save_path(
    smoke_inputs: SmokeInputs, method_name: str, tmp_path: Path
) -> None:
    options = METHOD_OPTIONS[method_name]
    pretrained_model_name = options.get("pretrained_model_name")
    pretrained_model_bundle_path = (
        None
        if pretrained_model_name is None
        else _pretrained_model_bundle_path(
            smoke_inputs,
            str(pretrained_model_name),
        )
    )
    save_path = tmp_path / "method.bundle"

    _run_script(
        f"train_method/{options['train_script_name']}.py",
        _load_renderer().render_training_contract(
            smoke_inputs.training_values(
                save_path, pretrained_model_bundle_path
            )
        ),
        " ".join(options.get("method_specific_params", [])),  # pyright: ignore[reportArgumentType, reportCallIssue]
        tmp_path / "hydra",
    )

    afa_method = _load_bundle_on_cpu(save_path)

    assert afa_method is not None

    if options["train_script_name"] in {"aaco", "aaco_nn"}:
        metadata = json.loads((save_path / "manifest.json").read_text())[
            "metadata"
        ]
        assert metadata["stage"] == "training"
        assert metadata["contract"]["dataset_key"] == DATASET_KEY
        assert metadata["contract"]["seed"] == SEED
