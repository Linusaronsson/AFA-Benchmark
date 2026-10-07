"""
Restore a real smoke run's native bundles and use them again.

Unlike `submission_harness`, which stubs every script, this runs the real
pipeline on CUBE with one dataset realization, `random_dummy` (no pretraining
stage) and `gdfs` (a pretrained model and the external classifier), saves
a smoke snapshot and restores it into a fresh workspace through the
public `scripts/release/snapshot.py` commands.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import pytest
import torch
import yaml
from typer.testing import CliRunner

from afabench.core.bundle_system.bundle import load_bundle
from afabench.release.manifest import (
    ExecutionMode,
    PayloadCategory,
    ReleaseManifest,
    ReleaseScope,
    read_release_manifest,
)
from scripts.release.snapshot import app

if TYPE_CHECKING:
    from afabench.core.types import AFADataset

REPO_ROOT = Path(__file__).parents[2]
PROFILE_CONFIGFILES = [
    REPO_ROOT / path
    for path in yaml.safe_load(
        (
            REPO_ROOT / "extra/workflow/profiles/config/all/config.yaml"
        ).read_text()
    )["configfile"]
]
SMOKE_SELECTION: dict[str, Any] = {
    "datasets": ["cube"],
    "dataset_realization_indices": [0],
    "methods": ["random_dummy", "gdfs"],
    "eval_hard_budgets": {"cube": [2]},
    "soft_budget_params": {
        "random_dummy": {"cube": [[0.3, None]]},
        "gdfs": {"cube": []},
    },
    "use_wandb": False,
    "smoke_test": True,
}
TAG = "initializer-cold"
GDFS_METHOD = (
    f"trained_methods/{TAG}/gdfs/dataset-cube+realization_index-0/"
    "pretrain_seed-0/"
    "train_seed-0+train_hard_budget-2+train_soft_budget_param-null/"
    "method.bundle"
)
GDFS_RAW_TABLE = (
    f"eval_results/eval_split-test/{TAG}/gdfs/dataset-cube+realization_index-0/"
    "pretrain_seed-0/"
    "train_seed-0+train_hard_budget-2+train_soft_budget_param-null/"
    "eval_seed-0+eval_hard_budget-2+eval_soft_budget_param-null/"
    "eval_data.parquet"
)
CPU = torch.device("cpu")


def workspace(root: Path) -> Path:
    """Make a checkout-shaped directory whose `extra/output` starts empty."""
    (root / "extra").mkdir(parents=True)
    for path in ["scripts", "extra/conf", "extra/data", "extra/workflow"]:
        (root / path).symlink_to(REPO_ROOT / path)
    return root


def snakemake(
    root: Path, configfiles: list[Path], *arguments: str
) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    environment.pop("SNAKEMAKE_PROFILE", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "--snakefile",
            "extra/workflow/snakefiles/orchestration/pipeline.smk",
            "--configfile",
            *(str(path) for path in configfiles),
            "--cores",
            "4",
            *arguments,
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=1800,
        check=False,
    )


def load_native(path: Path, category: PayloadCategory) -> object:
    """Load a bundle the way the training inputs and the evaluator do."""
    if category is PayloadCategory.DATASET_BUNDLE:
        return load_bundle(path)[0]
    return load_bundle(path, device=CPU)[0]


@pytest.fixture(scope="module")
def restored(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, list[Path], ReleaseManifest]:
    """Completed smoke workspace, restored workspace, config, manifest."""
    tmp_path = tmp_path_factory.mktemp("native-bundles")
    selection = tmp_path / "smoke.yaml"
    selection.write_text(yaml.safe_dump(SMOKE_SELECTION))
    configfiles = [*PROFILE_CONFIGFILES, selection]
    completed = workspace(tmp_path / "completed")
    run = snakemake(completed, configfiles, "all")
    assert run.returncode == 0, run.stdout + run.stderr

    runner = CliRunner()
    snapshot_dir = tmp_path / "snapshot"
    save = runner.invoke(
        app,
        [
            "save",
            str(snapshot_dir),
            "--source-root",
            str(completed / "extra/output"),
            *(
                argument
                for path in configfiles
                for argument in ["--configfile", str(path)]
            ),
            "--release-id",
            "smoke-native-bundles",
            "--scope",
            "smoke",
            "--checkout",
            str(REPO_ROOT),
        ],
    )
    assert save.exit_code == 0, save.output
    fresh = workspace(tmp_path / "fresh")
    restore = runner.invoke(
        app,
        [
            "restore",
            str(snapshot_dir),
            "--destination-root",
            str(fresh / "extra/output"),
        ],
    )
    assert restore.exit_code == 0, restore.output
    assert "Unreviewed dataset redistribution: cube" in restore.output
    manifest = read_release_manifest(fresh / "extra/release_manifest.json")
    return completed, fresh, configfiles, manifest


@pytest.mark.pipeline
def test_smoke_release_covers_every_payload_category(
    restored: tuple[Path, Path, list[Path], ReleaseManifest],
) -> None:
    _, _, _, manifest = restored

    assert manifest.scope is ReleaseScope.SMOKE
    assert manifest.execution_mode is ExecutionMode.SMOKE
    for payload in manifest.coverage.payloads:
        assert payload.scheduled > 0, payload
        assert payload.present == payload.scheduled, payload
        assert payload.size_bytes > 0, payload
    class_names = {
        payload.category: payload.class_names
        for payload in manifest.coverage.payloads
    }
    assert class_names[PayloadCategory.DATASET_BUNDLE] == ["CubeDataset"]
    assert class_names[PayloadCategory.CLASSIFIER_BUNDLE] == [
        "WrappedMaskedMLPClassifier"
    ]
    assert class_names[PayloadCategory.PRETRAINED_MODEL_BUNDLE] == [
        "GreedyAFAClassifier"
    ]
    assert class_names[PayloadCategory.AFA_METHOD_BUNDLE] == [
        "GDFSAFAMethod",
        "RandomWithoutClassifierAFAMethod",
    ]


@pytest.mark.pipeline
def test_smoke_release_retains_the_provenance_of_its_bundles(
    restored: tuple[Path, Path, list[Path], ReleaseManifest],
) -> None:
    _, _, _, manifest = restored
    bundles = {bundle.path: bundle for bundle in manifest.bundles}

    assert manifest.code.commit is not None
    assert manifest.settings.unmaskers == {"cube": "direct"}
    assert manifest.settings.initializer == "cold"
    assert manifest.settings.forcing_policy == (
        "forced_acquisition_when_eval_hard_budget_is_set"
    )
    dataset = bundles["datasets/cube/0/test.bundle"]
    assert dataset.bundle_manifest is not None
    generation = dataset.bundle_manifest["metadata"]
    assert generation["dataset_realization_index"] == 0
    assert generation["kwargs"]["seed"] == 0
    classifier = bundles[f"trained_classifiers/{TAG}/dataset-cube.bundle"]
    assert classifier.method_name is None
    assert classifier.seed == 0
    method = bundles[GDFS_METHOD]
    assert method.pretrained_model_name == "gdfs"
    assert method.train_hard_budget == 2
    assert method.bundle_manifest is not None
    contract = method.bundle_manifest["metadata"]["contract"]
    assert contract["smoke_test"] is True
    assert contract["seed"] == 0
    assert contract["unmasker"]["class_name"] == "DirectUnmasker"
    assert contract["initializer"]["class_name"] == "RandomInitializer"
    assert {
        (bundle_input.role, bundle_input.path)
        for bundle_input in method.inputs
    } == {
        ("train_dataset", "datasets/cube/0/train.bundle"),
        ("val_dataset", "datasets/cube/0/val.bundle"),
        ("classifier", classifier.path),
        (
            "pretrained_model",
            f"pretrained_models/{TAG}/gdfs/dataset-cube+realization_index-0/"
            "pretrain_seed-0/model.bundle",
        ),
    }


@pytest.mark.pipeline
def test_restored_bundles_load_through_the_native_loaders(
    restored: tuple[Path, Path, list[Path], ReleaseManifest],
) -> None:
    completed, fresh, _, manifest = restored

    for bundle in manifest.bundles:
        loaded = load_native(
            fresh / "extra/output" / bundle.path, bundle.category
        )
        assert bundle.bundle_manifest is not None
        assert type(loaded).__name__ == bundle.bundle_manifest["class_name"]
    original_test_split = cast(
        "AFADataset",
        load_native(
            completed / "extra/output/datasets/cube/0/test.bundle",
            PayloadCategory.DATASET_BUNDLE,
        ),
    )
    restored_test_split = cast(
        "AFADataset",
        load_native(
            fresh / "extra/output/datasets/cube/0/test.bundle",
            PayloadCategory.DATASET_BUNDLE,
        ),
    )
    assert len(restored_test_split) == len(original_test_split)
    for index in [0, len(original_test_split) - 1]:
        for restored_tensor, original_tensor in zip(
            restored_test_split[index], original_test_split[index], strict=True
        ):
            assert torch.equal(restored_tensor, original_tensor)


@pytest.mark.pipeline
def test_restored_bundles_reproduce_a_smoke_evaluation(
    restored: tuple[Path, Path, list[Path], ReleaseManifest],
) -> None:
    completed, fresh, _, _ = restored
    output = "extra/output"
    save_path = fresh / "rerun/eval_data.parquet"

    evaluation = subprocess.run(
        [
            sys.executable,
            "scripts/eval/eval_afa_method.py",
            f"method_bundle_path={output}/{GDFS_METHOD}",
            "initializer=cold",
            "unmasker=direct",
            f"dataset_bundle_path={output}/datasets/cube/0/test.bundle",
            f"save_path={save_path}",
            f"classifier_bundle_path={output}/trained_classifiers/{TAG}/"
            "dataset-cube.bundle",
            "seed=0",
            "device=cpu",
            "hard_budget=2",
            "soft_budget_param=null",
            "batch_size=32",
            "use_wandb=False",
            "smoke_test=True",
        ],
        cwd=fresh,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )

    assert evaluation.returncode == 0, evaluation.stdout + evaluation.stderr
    pd.testing.assert_frame_equal(
        pd.read_parquet(save_path),
        pd.read_parquet(completed / "extra/output" / GDFS_RAW_TABLE),
    )


@pytest.mark.pipeline
def test_snakemake_retrains_only_a_missing_bundle_from_restored_inputs(
    restored: tuple[Path, Path, list[Path], ReleaseManifest],
) -> None:
    _, fresh, configfiles, _ = restored
    method_bundle = fresh / "extra/output" / GDFS_METHOD
    for path in sorted(method_bundle.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    method_bundle.rmdir()
    target = f"extra/output/{GDFS_METHOD}"

    plan = snakemake(fresh, configfiles, "--dry-run", target)
    run = snakemake(fresh, configfiles, target)

    planned = plan.stdout + plan.stderr
    assert plan.returncode == 0, planned
    stats = planned.split("Job stats:", 1)[1].split("Reasons:", 1)[0]
    jobs = dict(re.findall(r"^(\w+)\s+(\d+)$", stats, flags=re.MULTILINE))
    assert jobs == {"train_method": "1", "total": "1"}, planned
    assert run.returncode == 0, run.stdout + run.stderr
    loaded = load_native(method_bundle, PayloadCategory.AFA_METHOD_BUNDLE)
    assert type(loaded).__name__ == "GDFSAFAMethod"
