"""Extend a downloaded benchmark through the single-config launcher path."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from afabench.release.manifest import (
    read_release_manifest,
    write_release_manifest,
)
from scripts.release.snapshot import app
from test.scripts.test_release_publishing import prepare_package
from test.workflow.submission_harness import WorkflowHarness
from test.workflow.test_native_bundle_restore import (
    GDFS_RAW_TABLE,
    REPO_ROOT,
    ReleaseManifest,
    restored,  # noqa: F401
    workspace,
)


def test_preview_exemption_cannot_execute_jobs(tmp_path: Path) -> None:
    harness = WorkflowHarness(tmp_path / "checkout")
    package = prepare_package(tmp_path, "preview-safety")
    manifest = read_release_manifest(package / "release_manifest.json")
    write_release_manifest(
        replace(
            manifest, bundles=[], evaluations=[], omitted_dataset_bundles=[]
        ),
        harness.output / "release_manifest.json",
    )
    executed = harness.run(
        "--executor",
        "local",
        "--config",
        "reuse_release=true",
        "release_preview=true",
    )
    assert executed.returncode != 0
    assert (
        "only valid in a dry run or estimate"
        in executed.stdout + executed.stderr
    )
    assert not harness.arguments.exists()


@pytest.mark.pipeline
def test_config_invocation_prepares_omitted_data_and_extends_realizations(
    restored: tuple[Path, Path, list[Path], ReleaseManifest],  # noqa: F811
    tmp_path: Path,
) -> None:
    completed, _, _, _ = restored
    package = tmp_path / "package"
    unreviewed = tmp_path / "unreviewed"
    policy = unreviewed / "conf/release/dataset_redistribution.yaml"
    policy.parent.mkdir(parents=True)
    review = yaml.safe_load(
        (REPO_ROOT / "conf/release/dataset_redistribution.yaml").read_text()
    )["datasets"]["cube"]
    policy.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "cube": {
                        **review,
                        "bundle_policy": "omit",
                        "omission_reason": "test omitted dataset",
                    }
                }
            }
        )
    )
    runner = CliRunner()
    saved = runner.invoke(
        app,
        [
            "save",
            str(package),
            "--source-root",
            str(completed / "output/smoke"),
            "--release-id",
            "single-config",
            "--checkout",
            str(unreviewed),
        ],
    )
    assert saved.exit_code == 0, saved.output
    fresh = workspace(tmp_path / "adopter")
    output = fresh / "output/smoke"
    restored_result = runner.invoke(
        app,
        [
            "restore",
            str(package),
            "--destination-root",
            str(output),
        ],
    )
    assert restored_result.exit_code == 0, restored_result.output
    assert not (output / "datasets/cube/0/train.bundle").exists()
    configfile = output / "benchmark.yaml"
    original = {
        path: path.stat().st_mtime_ns
        for folder in [
            "trained_classifiers",
            "pretrained_models",
            "trained_methods",
            "eval_results",
            "eval_results_transformed",
        ]
        for path in (output / folder).rglob("*")
        if path.is_file()
    }
    environment = {
        **os.environ,
        "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
    }
    environment.pop("SNAKEMAKE_PROFILE", None)
    invocation = [
        sys.executable,
        str(REPO_ROOT / "workflow/run_from_config.py"),
        "--configfile",
        str(configfile),
        "--jobs",
        "4",
        "--target",
        "all_eval_methods",
        "dataset_realization_indices=[0,1]",
        "list_jobs=true",
        "execution_file=workflow/profiles/execution/cpu.yaml",
    ]

    def run(*extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*invocation, *extra],
            cwd=fresh,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )

    preview = run("-n")
    text = preview.stdout + preview.stderr
    assert preview.returncode == 0, text
    assert "Release reuse preview" in text
    assert "realization_index=1" in text
    assert not (output / "datasets/cube/0/train.bundle").exists()
    executed = run()
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert (output / "datasets/cube/1/train.bundle").is_dir()
    assert (
        output
        / GDFS_RAW_TABLE.replace("realization_index-0", "realization_index-1")
        .replace("pretrain_seed-0", "pretrain_seed-1")
        .replace("train_seed-0", "train_seed-1")
        .replace("eval_seed-0", "eval_seed-1")
    ).is_file()
    assert {path: path.stat().st_mtime_ns for path in original} == original
    resumed = run()
    text = resumed.stdout + resumed.stderr
    assert resumed.returncode == 0, text
    assert "rule eval_method:" not in text
    assert "rule train_method:" not in text
    assert {path: path.stat().st_mtime_ns for path in original} == original

    data = next((output / "datasets/cube/0/train.bundle/data").rglob("*.pt"))
    data.write_bytes(data.read_bytes() + b"changed")
    incompatible = run()
    assert incompatible.returncode != 0
    assert "Incompatible dataset" in incompatible.stdout + incompatible.stderr
    assert {path: path.stat().st_mtime_ns for path in original} == original
