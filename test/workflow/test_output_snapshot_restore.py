"""Snapshot round trip through the real orchestration's output root."""

from pathlib import Path

import pytest

from afabench.release.snapshot import restore_snapshot, save_snapshot
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
