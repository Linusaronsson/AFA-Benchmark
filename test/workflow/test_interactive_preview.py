"""
`--interactive-preview` browses the jobs a real invocation schedules (#158).

The preview plans in a child process, `python -m afabench.pipeline_plan`,
which these tests run on the harness's workflow, and the application is
driven headless over that real plan.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

from textual.widgets import Static, Tree

from afabench.pipeline_plan.plan_json import load_plan
from afabench.pipeline_preview.app import PipelinePreviewApp
from afabench.pipeline_preview.planning_process import PlanningProcess
from test.workflow.submission_harness import WorkflowHarness


def plan_in_a_child_process(
    workflow: WorkflowHarness, arguments: list[str], output: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "afabench.pipeline_plan",
            "--output",
            str(output),
            "--",
            *arguments,
        ],
        cwd=workflow.root,
        env=workflow.environment(),
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )


def test_the_planning_child_writes_the_invocations_plan_as_json(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    output = tmp_path / "plan.json"

    planned = plan_in_a_child_process(
        workflow,
        workflow.invocation_arguments("--executor", "local"),
        output,
    )

    assert planned.returncode == 0, planned.stdout + planned.stderr
    plan = load_plan(output.read_text())
    # The harness's datasets and classifiers exist, so training and
    # evaluation remain.
    assert sorted(
        (job.rule, job.wildcards["method"]) for job in plan.scheduled_jobs
    ) == [
        ("eval_method", "alpha"),
        ("eval_method", "beta"),
        ("train_method", "alpha"),
        ("train_method", "beta"),
    ]
    assert workflow.submissions() == []


def test_the_planning_child_fails_with_the_invocations_diagnostics(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution"] = {"defaults": {"training": "tpu"}}
    output = tmp_path / "plan.json"

    planned = plan_in_a_child_process(
        workflow,
        workflow.invocation_arguments("--executor", "local"),
        output,
    )

    assert planned.returncode == 1
    assert "Invalid execution choice 'tpu'" in planned.stdout + planned.stderr
    assert not output.exists()


def test_the_preview_browses_the_jobs_the_workflow_schedules(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    app = PipelinePreviewApp(
        PlanningProcess(
            workflow.invocation_arguments("--executor", "local"),
            cwd=workflow.root,
            env=workflow.environment(),
        )
    )
    seen: dict[str, object] = {}

    async def browse() -> None:
        async with app.run_test(size=(120, 30)) as pilot:
            seen["opened with"] = str(app.query_one("#status", Static).content)
            deadline = time.monotonic() + 240
            while (
                app.planning_state == "planning"
                and time.monotonic() < deadline
            ):
                await pilot.pause(0.2)
            seen["diagnostics"] = app.diagnostics
            tree = app.query_one(Tree)
            seen["stages"] = [str(node.label) for node in tree.root.children]
            await pilot.press(
                "down", "right", "down", "right", "down", "right"
            )
            seen["jobs"] = (
                [str(node.label) for node in tree.cursor_node.children]
                if tree.cursor_node
                else []
            )
            await pilot.press("q")

    asyncio.run(browse())

    assert "Planning" in str(seen["opened with"])
    assert app.planning_state == "planned", seen["diagnostics"]
    assert seen["stages"] == ["training (2)", "evaluation (2)"]
    assert seen["jobs"] == [
        "train_method: realization 0, train seed 0, train hard budget 1"
    ]
    assert app.return_code == 0
    assert workflow.submissions() == []
    assert not workflow.arguments.exists()
