import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from afabench.release.manifest import (
    ReleaseScope,
    read_release_manifest,
)
from scripts.release.snapshot import app

runner = CliRunner()

RAW_HARD_BUDGET_TABLE = (
    "eval_results/eval_split-test/initializer-cold/alpha/"
    "dataset-cube+instance_idx-0/NO_PRETRAIN/"
    "train_seed-0+train_hard_budget-3+train_soft_budget_param-null/"
    "eval_seed-0+eval_hard_budget-3+eval_soft_budget_param-null/"
    "eval_data.parquet"
)


def workflow_config(*, smoke_test: bool) -> dict[str, Any]:
    return {
        "pretrain_mapping": {},
        "method_options": {
            "alpha": {"train_script_name": "alpha", "eval_batch_size": 4}
        },
        "methods": ["alpha"],
        "datasets": ["cube"],
        "dataset_instance_indices": [0, 1],
        "unmaskers": {"default": "direct"},
        "eval_hard_budgets": {"default": [3]},
        "soft_budget_params": {"alpha": {"default": [[0.5, None]]}},
        "classifier_names": {"default": "masked_mlp_classifier"},
        "use_wandb": False,
        "smoke_test": smoke_test,
    }


def write_configfile(path: Path, *, smoke_test: bool) -> Path:
    path.write_text(yaml.safe_dump(workflow_config(smoke_test=smoke_test)))
    return path


def build_output_tree(root: Path) -> None:
    bundle = root / "datasets/cube/0/train.bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"bundle_version": 1}')


def write_raw_table(root: Path) -> None:
    """One episode with external predictions only, as the evaluator saves."""
    path = root / RAW_HARD_BUDGET_TABLE
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "episode_id": [0, 0],
            "step": [0, 1],
            "action_performed": [2, 0],
            "builtin_predicted_class": [None, None],
            "external_predicted_class": [1, 0],
            "true_class": [0, 0],
            "accumulated_cost": [1.0, 1.0],
            "forced_stop": [False, False],
            "eval_seed": [0, 0],
            "eval_hard_budget": [3.0, 3.0],
        }
    ).to_parquet(path, index=False)


def save_args(
    snapshot_dir: Path, source_root: Path, configfile: Path, *extra: str
) -> list[str]:
    return [
        "save",
        str(snapshot_dir),
        "--source-root",
        str(source_root),
        "--configfile",
        str(configfile),
        *extra,
    ]


def read_manifest(snapshot_dir: Path) -> dict[str, Any]:
    return json.loads((snapshot_dir / "release_manifest.json").read_text())


def save_release(
    tmp_path: Path, *extra: str, smoke_test: bool = False
) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    write_raw_table(source_root)
    configfile = write_configfile(tmp_path / "run.yaml", smoke_test=smoke_test)
    snapshot_dir = tmp_path / "snapshot"
    result = runner.invoke(
        app,
        save_args(
            snapshot_dir,
            source_root,
            configfile,
            "--release-id",
            "2026-10-cube",
            *extra,
        ),
    )
    assert result.exit_code == 0, result.output
    return snapshot_dir, configfile


def test_save_with_release_id_writes_manifest_beside_output(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    configfile = write_configfile(tmp_path / "run.yaml", smoke_test=False)
    snapshot_dir = tmp_path / "snapshot"

    result = runner.invoke(
        app,
        save_args(
            snapshot_dir,
            source_root,
            configfile,
            "--release-id",
            "2026-10-cube",
            "--scope",
            "partial",
        ),
    )

    assert result.exit_code == 0, result.output
    assert (snapshot_dir / "output/datasets/cube/0/train.bundle").is_dir()
    manifest = read_manifest(snapshot_dir)
    assert manifest["manifest_version"] == 1
    assert manifest["release_id"] == "2026-10-cube"
    assert manifest["scope"] == "partial"
    assert manifest["execution_mode"] == "production"
    assert str(configfile) in result.output


def test_save_without_release_id_writes_no_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    snapshot_dir = tmp_path / "snapshot"

    result = runner.invoke(
        app, ["save", str(snapshot_dir), "--source-root", str(source_root)]
    )

    assert result.exit_code == 0, result.output
    assert not (snapshot_dir / "release_manifest.json").exists()


def test_manifest_options_without_release_id_are_refused(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    configfile = write_configfile(tmp_path / "run.yaml", smoke_test=False)
    snapshot_dir = tmp_path / "snapshot"

    result = runner.invoke(
        app, save_args(snapshot_dir, source_root, configfile)
    )

    assert result.exit_code != 0
    assert not snapshot_dir.exists()


@pytest.mark.parametrize("scope", ["full", "partial"])
def test_smoke_outputs_cannot_be_declared_a_release(
    tmp_path: Path, scope: str
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    configfile = write_configfile(tmp_path / "run.yaml", smoke_test=True)
    snapshot_dir = tmp_path / "snapshot"

    result = runner.invoke(
        app,
        save_args(
            snapshot_dir,
            source_root,
            configfile,
            "--release-id",
            "smoke",
            "--scope",
            scope,
        ),
    )

    assert result.exit_code != 0
    assert "smoke" in str(result.exception)
    assert not snapshot_dir.exists()


def test_smoke_override_on_production_config_is_also_refused(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    configfile = write_configfile(tmp_path / "run.yaml", smoke_test=False)
    snapshot_dir = tmp_path / "snapshot"

    result = runner.invoke(
        app,
        save_args(
            snapshot_dir,
            source_root,
            configfile,
            "--config",
            "smoke_test=true",
            "--release-id",
            "smoke",
            "--scope",
            "full",
        ),
    )

    assert result.exit_code != 0
    assert not snapshot_dir.exists()


def test_smoke_outputs_are_recorded_as_smoke(tmp_path: Path) -> None:
    snapshot_dir, _ = save_release(
        tmp_path, "--scope", "smoke", smoke_test=True
    )

    manifest = read_manifest(snapshot_dir)
    assert manifest["scope"] == "smoke"
    assert manifest["execution_mode"] == "smoke"


def test_manifest_records_each_configfile_and_override(
    tmp_path: Path,
) -> None:
    snapshot_dir, configfile = save_release(
        tmp_path,
        "--scope",
        "partial",
        "--config",
        "initializer=warm",
        "--config",
        "datasets=[cube]",
    )

    workflow = read_manifest(snapshot_dir)["workflow_config"]
    assert workflow["profile"] is None
    assert [record["path"] for record in workflow["configfiles"]] == [
        str(configfile)
    ]
    assert len(workflow["configfiles"][0]["sha256"]) == 64
    assert workflow["overrides"] == {
        "initializer": "warm",
        "datasets": ["cube"],
    }
    assert workflow["merged"]["initializer"] == "warm"
    assert workflow["merged"]["methods"] == ["alpha"]


def test_profile_supplies_configfiles_and_cli_config_replaces_its_config(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    configfile = write_configfile(tmp_path / "run.yaml", smoke_test=False)
    profile = tmp_path / "profile"
    profile.mkdir()
    (profile / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "configfile": [str(configfile)],
                "config": {"initializer": "warm", "use_wandb": True},
            }
        )
    )
    snapshot_dir = tmp_path / "snapshot"

    result = runner.invoke(
        app,
        [
            "save",
            str(snapshot_dir),
            "--source-root",
            str(source_root),
            "--profile",
            str(profile),
            "--config",
            "initializer=random",
            "--release-id",
            "2026-10-cube",
            "--scope",
            "partial",
        ],
    )

    assert result.exit_code == 0, result.output
    assert str(profile) in result.output
    manifest = read_manifest(snapshot_dir)
    workflow = manifest["workflow_config"]
    assert workflow["profile"] == str(profile)
    assert [record["path"] for record in workflow["configfiles"]] == [
        str(configfile)
    ]
    assert workflow["overrides"] == {"initializer": "random"}
    assert workflow["merged"]["use_wandb"] is False
    assert manifest["settings"]["initializer"] == "random"


def test_manifest_records_resolved_settings(tmp_path: Path) -> None:
    snapshot_dir, _ = save_release(tmp_path, "--scope", "partial")

    settings = read_manifest(snapshot_dir)["settings"]
    assert settings["initializer"] == "cold"
    assert settings["eval_split"] == "test"
    assert settings["dataset_instance_indices"] == [0, 1]
    assert settings["dataset_splits"] == ["train", "val", "test"]
    assert settings["unmaskers"] == {"cube": "direct"}
    assert settings["feature_costs"] == {
        "cube": {"path": None, "sha256": None}
    }
    assert settings["eval_batch_sizes"] == {"alpha": {"cube": 4}}
    assert settings["classifiers"] == [
        {
            "bundle_path": (
                "trained_classifiers/initializer-cold/dataset-cube.bundle"
            ),
            "script_name": "masked_mlp_classifier",
            "script_params": "",
            "dataset_key": "cube",
            "method_name": None,
            "dataset_instance_index": 0,
            "seed": 0,
        }
    ]


def test_manifest_records_feature_cost_file_of_the_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    costs = checkout / "extra/data/misc/feature_costs/cube.csv"
    costs.parent.mkdir(parents=True)
    costs.write_text("1,2,3\n")

    snapshot_dir, _ = save_release(
        tmp_path, "--scope", "partial", "--checkout", str(checkout)
    )

    assert read_manifest(snapshot_dir)["settings"]["feature_costs"] == {
        "cube": {
            "path": "extra/data/misc/feature_costs/cube.csv",
            "sha256": (
                "7a8988e95e356e2b5b8fecf5e31f7c2e"
                "7e8fb44a5cd9d89ebb0d1e60b1f5c689"
            ),
        }
    }


def test_manifest_lists_every_configured_table_with_its_identity(
    tmp_path: Path,
) -> None:
    snapshot_dir, _ = save_release(tmp_path, "--scope", "partial")

    tables = read_manifest(snapshot_dir)["evaluation_tables"]
    assert len(tables) == 4  # two instances, one hard and one soft budget
    hard = next(t for t in tables if t["raw_path"] == RAW_HARD_BUDGET_TABLE)
    assert hard == {
        "raw_path": RAW_HARD_BUDGET_TABLE,
        "transformed_path": RAW_HARD_BUDGET_TABLE.replace(
            "eval_results/", "eval_results_transformed/", 1
        ),
        "raw_present": True,
        "transformed_present": False,
        "raw_size_bytes": (tmp_path / "source" / RAW_HARD_BUDGET_TABLE)
        .stat()
        .st_size,
        "transformed_size_bytes": None,
        "method_name": "alpha",
        "dataset_key": "cube",
        "dataset_instance_index": 0,
        "dataset_generation_seed": 0,
        "eval_split": "test",
        "initializer": "cold",
        "unmasker": "direct",
        "budget_setting": "hard_budget",
        "pretrained_model_name": None,
        "pretrain_seed": None,
        "train_seed": 0,
        "train_hard_budget": 3,
        "train_soft_budget_param": None,
        "eval_seed": 0,
        "eval_hard_budget": 3,
        "eval_soft_budget_param": None,
        "forced_acquisition": True,
        "classifier_bundle_path": (
            "trained_classifiers/initializer-cold/dataset-cube.bundle"
        ),
        "eval_batch_size": 4,
        "classifier_variants": ["external"],
        "inputs": [
            {"role": "eval_dataset", "path": "datasets/cube/0/test.bundle"},
            {
                "role": "method",
                "path": (
                    "trained_methods/initializer-cold/alpha/"
                    "dataset-cube+instance_idx-0/NO_PRETRAIN/"
                    "train_seed-0+train_hard_budget-3+"
                    "train_soft_budget_param-null/method.bundle"
                ),
            },
            {
                "role": "classifier",
                "path": (
                    "trained_classifiers/initializer-cold/dataset-cube.bundle"
                ),
            },
        ],
    }
    soft = next(
        t
        for t in tables
        if t["dataset_instance_index"] == 1
        and t["budget_setting"] == "soft_budget"
    )
    assert soft["raw_present"] is False
    assert soft["train_soft_budget_param"] == 0.5
    assert soft["eval_hard_budget"] is None
    assert soft["forced_acquisition"] is False
    assert soft["classifier_variants"] is None


def test_coverage_describes_only_the_outputs_present(tmp_path: Path) -> None:
    snapshot_dir, _ = save_release(tmp_path, "--scope", "partial")

    coverage = read_manifest(snapshot_dir)["coverage"]
    # Per-category payload counts are pinned in test_release_native_payloads.
    del coverage["payloads"]
    assert coverage == {
        "datasets": ["cube"],
        "dataset_instance_indices": [0],
        "methods": ["alpha"],
        "eval_splits": ["test"],
        "budget_settings": ["hard_budget"],
        "classifier_variants": ["external"],
        "output_categories": ["datasets", "eval_results"],
    }


def test_restore_keeps_manifest_beside_the_restored_root(
    tmp_path: Path,
) -> None:
    snapshot_dir, _ = save_release(tmp_path, "--scope", "partial")
    destination_root = tmp_path / "checkout/extra/output"

    result = runner.invoke(
        app,
        [
            "restore",
            str(snapshot_dir),
            "--destination-root",
            str(destination_root),
        ],
    )

    assert result.exit_code == 0, result.output
    restored = tmp_path / "checkout/extra/release_manifest.json"
    assert str(restored) in result.output
    assert "2026-10-cube" in result.output
    assert (
        restored.read_bytes()
        == (snapshot_dir / "release_manifest.json").read_bytes()
    )
    assert (destination_root / RAW_HARD_BUDGET_TABLE).is_file()
    manifest = read_release_manifest(restored)
    assert manifest.scope is ReleaseScope.PARTIAL
    assert manifest.evaluation_tables[0].method_name == "alpha"


def test_restore_refuses_an_existing_manifest_and_restores_nothing(
    tmp_path: Path,
) -> None:
    snapshot_dir, _ = save_release(tmp_path, "--scope", "partial")
    destination_root = tmp_path / "checkout/extra/output"
    existing = tmp_path / "checkout/extra/release_manifest.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("pre-existing")

    result = runner.invoke(
        app,
        [
            "restore",
            str(snapshot_dir),
            "--destination-root",
            str(destination_root),
        ],
    )

    assert result.exit_code != 0
    assert existing.read_text() == "pre-existing"
    assert not destination_root.exists()


def test_restore_refuses_an_unknown_manifest_version(tmp_path: Path) -> None:
    snapshot_dir, _ = save_release(tmp_path, "--scope", "partial")
    manifest_path = snapshot_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["manifest_version"] = 99
    manifest_path.write_text(json.dumps(manifest))
    destination_root = tmp_path / "checkout/extra/output"

    result = runner.invoke(
        app,
        [
            "restore",
            str(snapshot_dir),
            "--destination-root",
            str(destination_root),
        ],
    )

    assert result.exit_code != 0
    assert "99" in str(result.exception)
    assert not destination_root.exists()


def test_save_refuses_an_existing_manifest_and_copies_nothing(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    configfile = write_configfile(tmp_path / "run.yaml", smoke_test=False)
    snapshot_dir = tmp_path / "snapshot"
    existing = snapshot_dir / "release_manifest.json"
    existing.parent.mkdir()
    existing.write_text("pre-existing")

    result = runner.invoke(
        app,
        save_args(
            snapshot_dir,
            source_root,
            configfile,
            "--release-id",
            "2026-10-cube",
            "--scope",
            "partial",
        ),
    )

    assert result.exit_code != 0
    assert existing.read_text() == "pre-existing"
    assert not (snapshot_dir / "output").exists()


def git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_manifest_records_commit_and_dirty_state_of_the_checkout(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    git(checkout, "init", "--quiet")
    (checkout / "code.py").write_text("x = 1\n")
    git(checkout, "add", "code.py")
    git(
        checkout,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )
    commit = git(checkout, "rev-parse", "HEAD")
    (checkout / "untracked.txt").write_text("ignored")

    clean_dir, _ = save_release(
        tmp_path / "clean", "--scope", "partial", "--checkout", str(checkout)
    )
    (checkout / "code.py").write_text("x = 2\n")
    dirty_dir, _ = save_release(
        tmp_path / "dirty", "--scope", "partial", "--checkout", str(checkout)
    )

    assert read_manifest(clean_dir)["code"] == {
        "commit": commit,
        "dirty": False,
    }
    assert read_manifest(dirty_dir)["code"] == {
        "commit": commit,
        "dirty": True,
    }


def test_manifest_records_null_code_identity_outside_git(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "not-a-repo"
    checkout.mkdir()

    snapshot_dir, _ = save_release(
        tmp_path, "--scope", "partial", "--checkout", str(checkout)
    )

    assert read_manifest(snapshot_dir)["code"] == {
        "commit": None,
        "dirty": None,
    }
