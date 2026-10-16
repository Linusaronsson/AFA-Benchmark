"""The real workflow records its effective configuration only when executing."""

import json
from pathlib import Path

from afabench.release.workflow_config import (
    WORKFLOW_CONFIG_FILENAME,
    release_workflow_config,
)
from test.workflow.submission_harness import WorkflowHarness


def test_workflow_records_config_without_dry_runs_overwriting_it(
    tmp_path: Path,
) -> None:
    harness = WorkflowHarness(tmp_path)
    path = harness.output / WORKFLOW_CONFIG_FILENAME
    planned = harness.run("--executor", "local", "--dry-run")
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert not path.exists()

    executed = harness.run(
        "--executor", "local", "--config", "methods=[alpha]", "use_wandb=false"
    )
    assert executed.returncode == 0, executed.stdout + executed.stderr
    recorded = release_workflow_config(harness.output)
    assert recorded.merged["methods"] == ["alpha"]
    assert recorded.overrides == {"methods": ["alpha"], "use_wandb": "false"}
    assert (
        release_workflow_config(
            harness.output,
            configfiles=[tmp_path / "config.yaml"],
            overrides=["methods=[alpha]", "use_wandb=false"],
        )
        == recorded
    )
    assert recorded.merged["smoke_test"] is True
    assert recorded.configfiles
    assert len(recorded.configfiles[0].sha256) == 64
    before = path.read_bytes()

    harness.config["methods"] = ["beta"]
    planned = harness.run("--executor", "local", "--dry-run")
    assert planned.returncode == 0, planned.stdout + planned.stderr
    assert path.read_bytes() == before
    assert json.loads(before)["merged"]["methods"] == ["alpha"]
