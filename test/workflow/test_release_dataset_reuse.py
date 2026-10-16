"""Regenerated omitted datasets must not invalidate released artifacts."""

import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from scripts.release.snapshot import app
from test.workflow.test_native_bundle_restore import (
    GDFS_METHOD,
    ReleaseManifest,
    restored,  # noqa: F401
    snakemake,
    workspace,
)


@pytest.fixture
def regenerated_release(
    restored: tuple[Path, Path, list[Path], ReleaseManifest],  # noqa: F811
    tmp_path: Path,
) -> tuple[Path, list[Path]]:
    completed, _, configfiles, _ = restored
    package = tmp_path / "package"
    checkout = tmp_path / "unreviewed"
    checkout.mkdir()
    runner = CliRunner()
    saved = runner.invoke(
        app,
        [
            "save",
            str(package),
            "--source-root",
            str(completed / "output/smoke"),
            "--release-id",
            "omitted-dataset",
            "--scope",
            "partial",
            "--checkout",
            str(checkout),
            *(
                arg
                for path in configfiles
                for arg in ["--configfile", str(path)]
            ),
        ],
    )
    assert saved.exit_code == 0, saved.output
    fresh = workspace(tmp_path / "adopter")
    output = fresh / "output/smoke"
    restored_result = runner.invoke(
        app, ["restore", str(package), "--destination-root", str(output)]
    )
    assert restored_result.exit_code == 0, restored_result.output
    assert not (output / "datasets/cube/0/train.bundle").exists()
    generated = snakemake(fresh, configfiles, "all_generate_datasets")
    assert generated.returncode == 0, generated.stdout + generated.stderr
    return fresh, configfiles


@pytest.mark.pipeline
def test_regenerated_release_dataset_keeps_existing_models_and_evaluations(
    regenerated_release: tuple[Path, list[Path]], tmp_path: Path
) -> None:
    fresh, configfiles = regenerated_release
    output = fresh / "output/smoke"
    target = f"output/smoke/{GDFS_METHOD}"
    assert (
        output / "datasets/cube/0/train.bundle/.snakemake_timestamp"
    ).stat().st_mtime > (
        output / GDFS_METHOD / ".snakemake_timestamp"
    ).stat().st_mtime
    plan = snakemake(
        fresh,
        configfiles,
        "--dry-run",
        "all_eval_methods",
        "--config",
        "reuse_release=true",
        "stages={dataset_generation: false}",
    )
    text = plan.stdout + plan.stderr
    assert plan.returncode == 0, text
    assert "Nothing to be done" in text, text
    benchmark = snakemake(
        fresh,
        configfiles,
        "--dry-run",
        "all",
        "--config",
        "reuse_release=true",
        "stages={dataset_generation: false}",
        "list_jobs=true",
    )
    text = benchmark.stdout + benchmark.stderr
    assert benchmark.returncode == 0, text
    for rule in [
        "train_classifier",
        "pretrain_model",
        "train_method",
        "eval_method",
        "transform_eval_data",
    ]:
        assert f"rule {rule}:" not in text, text
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
    # A new configured method trains against the released prerequisites.
    variant = tmp_path / "variant.yaml"
    variant.write_text(
        yaml.safe_dump(
            {
                "methods": ["random_dummy", "gdfs", "gdfs_adopter"],
                "method_options": {
                    "gdfs_adopter": {
                        "train_script_name": "gdfs",
                        "pretrained_model_name": "gdfs",
                        "eval_batch_size": {"default": 32},
                    }
                },
                "soft_budget_params": {"gdfs_adopter": {"default": []}},
            }
        )
    )
    new_method = target.replace("/gdfs/", "/gdfs_adopter/")
    trained = snakemake(
        fresh,
        [*configfiles, variant],
        "all_eval_methods",
        "--config",
        "reuse_release=true",
        "stages={dataset_generation: false}",
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    assert (fresh / new_method).is_dir()
    assert {path: path.stat().st_mtime_ns for path in original} == original
    # Missing methods still train; released prerequisites stay reusable.
    shutil.rmtree(output / GDFS_METHOD)
    missing = snakemake(
        fresh,
        configfiles,
        "--dry-run",
        target,
        "--config",
        "reuse_release=true",
        "stages={dataset_generation: false}",
        "list_jobs=true",
    )
    text = missing.stdout + missing.stderr
    assert missing.returncode == 0, text
    assert "rule train_method:" in text
    assert "rule pretrain_model:" not in text
    assert "rule train_classifier:" not in text

    development = snakemake(
        fresh,
        configfiles,
        "--dry-run",
        "all_eval_methods",
        "--config",
        "list_jobs=true",
    )
    text = development.stdout + development.stderr
    assert development.returncode == 0, text
    assert "rule train_classifier:" in text

    # Hash the bytes, not a possibly stale content_hash in the manifest.
    data_file = next(
        (output / "datasets/cube/0/train.bundle/data").rglob("*.pt")
    )
    data_file.write_bytes(data_file.read_bytes() + b"changed")
    incompatible = snakemake(
        fresh,
        configfiles,
        "--dry-run",
        "all_eval_methods",
        "--config",
        "reuse_release=true",
        "stages={dataset_generation: false}",
    )
    assert incompatible.returncode != 0
    assert "Incompatible dataset" in incompatible.stdout + incompatible.stderr
    changed_development = snakemake(
        fresh,
        configfiles,
        "--dry-run",
        "all_eval_methods",
        "--config",
        "list_jobs=true",
    )
    text = changed_development.stdout + changed_development.stderr
    assert changed_development.returncode == 0, text
    assert "rule train_classifier:" in text

    unsafe = snakemake(
        fresh,
        configfiles,
        "--dry-run",
        "all_eval_methods",
        "--config",
        "reuse_release=true",
    )
    assert unsafe.returncode != 0
    assert (
        "requires stages={dataset_generation: false}"
        in unsafe.stdout + unsafe.stderr
    )
