"""
The preview browses the plan headless: panes, keys, layouts, failure (#158).

Driven through Textual's pilot over stand-in planning children, so no
Snakemake runs; `test/workflow/test_interactive_preview.py` browses a
real workflow's plan.
"""

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import Static, TextArea, Tree

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.planning import (
    JobStatus,
    PipelinePlan,
    PlannedJob,
    StatusReason,
)
from afabench.pipeline_preview.app import STATUS_STYLES, PipelinePreviewApp
from afabench.pipeline_preview.planning_process import PlanningProcess

ARGUMENTS = ["--profile", "workflow/profiles/pipeline/all", "all"]
# Wide enough for both panes; too narrow for them side by side.
WIDE = (120, 30)
NARROW = (60, 24)
PLANNING_TIMEOUT_SECONDS = 20


def browse(
    interpreter: list[str],
    scenario: Callable[[Pilot[None], PipelinePreviewApp], Awaitable[None]],
    *,
    size: tuple[int, int] = WIDE,
) -> PipelinePreviewApp:
    """Run the app over the interpreter's planning child; return it exited."""
    app = PipelinePreviewApp(
        PlanningProcess(ARGUMENTS, interpreter=interpreter)
    )

    async def run() -> None:
        async with app.run_test(size=size) as pilot:
            await scenario(pilot, app)

    asyncio.run(run())
    return app


async def planned(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
    """Wait until planning in the child ended."""
    deadline = time.monotonic() + PLANNING_TIMEOUT_SECONDS
    while app.planning_state == "planning" and time.monotonic() < deadline:
        await pilot.pause(0.05)
    assert app.planning_state != "planning", "planning did not end"


def status(app: PipelinePreviewApp) -> str:
    return str(app.query_one("#status", Static).content)


def details(app: PipelinePreviewApp) -> str:
    return str(app.query_one("#details-text", Static).content)


def focused(app: PipelinePreviewApp) -> str:
    return (app.focused.id if app.focused else None) or ""


def cursor_label(app: PipelinePreviewApp) -> str:
    node = app.query_one(Tree).cursor_node
    assert node is not None
    return str(node.label)


def test_opens_planning_and_quitting_ends_the_child(
    hanging_interpreter: tuple[list[str], Path],
) -> None:
    interpreter, pid_file = hanging_interpreter

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        assert "Planning" in status(app)
        while not pid_file.exists():
            await pilot.pause(0.05)
        await pilot.press("q")

    app = browse(interpreter, scenario)

    assert app.return_code == 0
    assert not alive(int(pid_file.read_text()))


def test_the_tree_counts_the_jobs_of_each_group_by_status(
    planning_interpreter: list[str],
) -> None:
    labels: list[str] = []

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        tree = app.query_one(Tree)
        labels.extend(str(node.label) for node in tree.root.children)
        assert "4 jobs: 3 Run, 1 Reuse" in status(app)
        assert not app.query_one("#release-warning").display
        await pilot.press("q")

    browse(planning_interpreter, scenario)

    assert labels == ["training (2 Run, 1 Reuse)", "evaluation (1 Run)"]


def test_jobs_wear_their_status_as_a_badge_and_a_color(
    planning_interpreter: list[str],
) -> None:
    seen: dict[str, object] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        # Expand training, alpha and cube: a run job, then a reused one.
        await pilot.press("down", "right", "down", "right", "down", "right")
        tree = app.query_one(Tree)
        assert tree.cursor_node is not None
        jobs = [node.label for node in tree.cursor_node.children]
        seen["labels"] = [str(label) for label in jobs]
        seen["styles"] = [
            {span.style for span in label.spans if isinstance(span.style, str)}
            for label in jobs
        ]
        await pilot.press("q")

    browse(planning_interpreter, scenario)

    assert seen["labels"] == [
        "Run train_method: realization 0, train hard budget 4",
        "Reuse train_method: realization 1, train hard budget 4",
    ]
    run_styles, reuse_styles = seen["styles"]
    assert run_styles
    assert reuse_styles
    assert run_styles != reuse_styles


def released_training(realization: int, status: JobStatus) -> PlannedJob:
    """Alpha's training on a realization of a release preview."""
    return PlannedJob(
        rule="train_method",
        wildcards={},
        identity=JobIdentity(
            stage="training",
            name="alpha",
            dataset_key="cube",
            dataset_realization_index=realization,
            train_hard_budget=4,
        ),
        device="cpu",
        cpus=2,
        gpus=0,
        jobid=realization,
        status=status,
        reasons=(
            StatusReason(kind="outputs_up_to_date")
            if status == "reuse"
            else StatusReason(
                kind="pending_dataset_check",
                files=(f"datasets/cube/{realization}/train.bundle",),
            ),
        ),
    )


@pytest.mark.parametrize(
    "plan",
    [
        PipelinePlan(
            jobs=[
                released_training(0, "conditional_reuse"),
                released_training(1, "reuse"),
            ]
        )
    ],
)
def test_conditional_reuse_wears_its_own_badge_below_the_release_warning(
    planning_interpreter: list[str],
) -> None:
    seen: dict[str, object] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        seen["status"] = status(app)
        warning = app.query_one("#release-warning", Static)
        seen["warning"] = str(warning.content) if warning.display else None
        # Expand training, alpha and cube.
        await pilot.press("down", "right", "down", "right", "down", "right")
        tree = app.query_one(Tree)
        assert tree.cursor_node is not None
        jobs = [node.label for node in tree.cursor_node.children]
        seen["labels"] = [str(label) for label in jobs]
        seen["styles"] = [
            {span.style for span in label.spans if isinstance(span.style, str)}
            for label in jobs
        ]
        await pilot.press("q")

    browse(planning_interpreter, scenario)

    assert seen["status"] == "2 jobs: 0 Run, 1 Reuse, 1 Conditional reuse"
    assert "regenerated datasets must match" in str(seen["warning"])
    assert seen["labels"] == [
        "Conditional reuse train_method: realization 0, train hard budget 4",
        "Reuse train_method: realization 1, train hard budget 4",
    ]
    conditional_styles, reuse_styles = seen["styles"]
    assert conditional_styles == {STATUS_STYLES["conditional_reuse"]}
    assert reuse_styles == {STATUS_STYLES["reuse"]}
    # Each status has a color of its own.
    assert len(set(STATUS_STYLES.values())) == len(STATUS_STYLES)


def test_arrows_walk_the_tree_and_enter_opens_the_details(
    planning_interpreter: list[str],
) -> None:
    seen: dict[str, str] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        await pilot.press("down")
        seen["group"] = cursor_label(app)
        seen["group details"] = details(app)
        # Expand training, alpha and cube, to the first job.
        await pilot.press("right", "down", "right", "down", "right", "down")
        seen["job"] = cursor_label(app)
        await pilot.press("enter")
        seen["focused"] = focused(app)
        seen["job details"] = details(app)
        await pilot.press("q")

    browse(planning_interpreter, scenario)

    assert seen["group"] == "training (2 Run, 1 Reuse)"
    assert "By method: alpha (1 Run, 1 Reuse), beta (1 Run)" in str(
        seen["group details"]
    )
    assert (
        seen["job"] == "Run train_method: realization 0, train hard budget 4"
    )
    assert seen["focused"] == "details"
    assert "Status: Run" in seen["job details"]
    assert "Reason: Missing outputs: train_method.out" in seen["job details"]
    assert "Training hard budget: 4" in seen["job details"]


def test_tab_switches_the_focused_pane(
    planning_interpreter: list[str],
) -> None:
    panes: list[str] = []

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        for _ in range(3):
            panes.append(focused(app))
            await pilot.press("tab")
        await pilot.press("q")

    browse(planning_interpreter, scenario)

    assert panes == ["tree", "details", "tree"]


def test_a_narrow_terminal_shows_one_pane_at_a_time(
    planning_interpreter: list[str],
) -> None:
    shown: list[tuple[bool, bool]] = []

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        tree, details_pane = app.query_one("#tree"), app.query_one("#details")
        shown.append((tree.display, details_pane.display))
        await pilot.press("tab")
        shown.append((tree.display, details_pane.display))
        await pilot.resize_terminal(*WIDE)
        shown.append((tree.display, details_pane.display))
        await pilot.press("q")

    browse(planning_interpreter, scenario, size=NARROW)

    assert shown == [(True, False), (False, True), (True, True)]


def test_a_planning_failure_stays_visible_and_exits_nonzero(
    failing_interpreter: list[str],
) -> None:
    seen: dict[str, object] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        seen["status"] = status(app)
        seen["diagnostics"] = app.query_one(TextArea).text
        seen["tree shown"] = app.query_one("#tree").display
        await pilot.press("c")
        seen["clipboard"] = app.clipboard
        await pilot.press("q")

    app = browse(failing_interpreter, scenario)

    assert "Planning failed" in str(seen["status"])
    assert "MissingInputException: cube/0/train.bundle" in str(
        seen["diagnostics"]
    )
    assert seen["clipboard"] == seen["diagnostics"]
    assert seen["tree shown"] is False
    assert app.return_code == 1


def alive(pid: int) -> bool:
    """Whether the process exists; a reaped child does not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
