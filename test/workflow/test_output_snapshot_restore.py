"""Snapshot round trip through the real orchestration's output root."""

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from afabench.release.manifest import (
    ClassifierVariant,
    ExecutionMode,
    ReleaseScope,
    read_release_manifest,
)
from afabench.release.snapshot import restore_snapshot, save_snapshot
from scripts.release.snapshot import app
from test.workflow.submission_harness import WorkflowHarness


@pytest.mark.pipeline
def test_restored_snapshot_leaves_nothing_for_a_dry_run_to_schedule(
    tmp_path: Path,
) -> None:
    completed = WorkflowHarness(tmp_path / "completed")
    run = completed.run("--executor", "local", target="all_eval_methods")
    assert run.returncode == 0, run.stdout + run.stderr

    snapshot_dir = tmp_path / "snapshot"
    save_snapshot(tmp_path / "completed/extra/output", snapshot_dir)

    fresh = WorkflowHarness(tmp_path / "fresh")
    restore_snapshot(snapshot_dir, tmp_path / "fresh/extra/output")

    result = fresh.run(
        "--dry-run", "--executor", "local", target="all_eval_methods"
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Nothing to be done" in output


# Stands in for the evaluator: writes a one-episode raw table with external
# predictions only, so the manifest can read its prediction columns.
FAKE_EVALUATOR = """\
import sys
from pathlib import Path

import pandas as pd

args = dict(arg.split("=", 1) for arg in sys.argv[1:])
out = Path(args["save_path"])
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    {
        "episode_id": [0],
        "step": [0],
        "action_performed": [0],
        "builtin_predicted_class": [None],
        "external_predicted_class": [1],
        "true_class": [1],
        "accumulated_cost": [0.0],
        "forced_stop": [False],
        "eval_seed": [0],
        "eval_hard_budget": [1.0],
    }
).to_parquet(out, index=False)
"""


@pytest.mark.pipeline
def test_smoke_snapshot_round_trips_its_test_only_manifest(
    tmp_path: Path,
) -> None:
    completed = WorkflowHarness(tmp_path / "completed")
    (tmp_path / "completed/scripts/eval/eval_afa_method.py").write_text(
        FAKE_EVALUATOR
    )
    run = completed.run("--executor", "local", target="all_eval_methods")
    assert run.returncode == 0, run.stdout + run.stderr
    configfile = tmp_path / "smoke.yaml"
    configfile.write_text(yaml.safe_dump(completed.config))
    snapshot_dir = tmp_path / "snapshot"
    runner = CliRunner()

    save = runner.invoke(
        app,
        [
            "save",
            str(snapshot_dir),
            "--source-root",
            str(tmp_path / "completed/extra/output"),
            "--configfile",
            str(configfile),
            "--release-id",
            "smoke-check",
            "--scope",
            "test_only",
        ],
    )
    assert save.exit_code == 0, save.output
    fresh = WorkflowHarness(tmp_path / "fresh")
    restore = runner.invoke(
        app,
        [
            "restore",
            str(snapshot_dir),
            "--destination-root",
            str(tmp_path / "fresh/extra/output"),
        ],
    )
    assert restore.exit_code == 0, restore.output

    manifest = read_release_manifest(
        tmp_path / "fresh/extra/release_manifest.json"
    )
    assert manifest.scope is ReleaseScope.TEST_ONLY
    assert manifest.execution_mode is ExecutionMode.SMOKE
    assert manifest.evaluation_tables
    assert all(table.raw_present for table in manifest.evaluation_tables)
    assert manifest.coverage.methods == ["alpha", "beta"]
    assert manifest.coverage.classifier_variants == [
        ClassifierVariant.EXTERNAL
    ]
    result = fresh.run(
        "--dry-run", "--executor", "local", target="all_eval_methods"
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "Nothing to be done" in output
