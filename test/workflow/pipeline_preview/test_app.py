"""
The preview browses the plan headless: panes, keys, layouts, failure (#158).

Driven through Textual's pilot over stand-in planning children, so no
Snakemake runs; `test/workflow/test_interactive_preview.py` browses a
real workflow's plan. In the workflow tier, as each test drives the app for
seconds, which the fast tier's budget cannot afford.
"""

import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path

import pytest
from textual.pilot import Pilot
from textual.widgets import OptionList, Static, TextArea, Tree

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.planning import (
    JobStatus,
    PipelinePlan,
    PlannedJob,
    StatusReason,
)
from afabench.pipeline_preview.app import STATUS_STYLES, PipelinePreviewApp
from afabench.pipeline_preview.model import (
    DEFAULT_VIEW,
    GROUPINGS,
    OUTSIDE_FILTERED_TREE,
    Dimension,
    PlanView,
)
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
    view: PlanView = DEFAULT_VIEW,
) -> PipelinePreviewApp:
    """Run the app over the interpreter's planning child; return it exited."""
    app = PipelinePreviewApp(
        PlanningProcess(ARGUMENTS, interpreter=interpreter), view=view
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


def top_groups(app: PipelinePreviewApp) -> list[str]:
    return [str(node.label) for node in app.query_one(Tree).root.children]


def moves(
    selected: tuple[Dimension, ...], chosen: tuple[Dimension, ...]
) -> list[str]:
    """Return the keys moving from a grouping to another in its options."""
    steps = GROUPINGS.index(chosen) - GROUPINGS.index(selected)
    return ["down"] * steps if steps > 0 else ["up"] * -steps


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
        tree = app.query_one("#tree")
        details_pane = app.query_one("#details-pane")
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


def test_the_grouping_control_regroups_the_tree(
    planning_interpreter: list[str],
) -> None:
    seen: dict[str, list[str]] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        seen["initial"] = top_groups(app)
        # Choose method, then dataset → method.
        await pilot.press("g", "enter", *moves(("dataset",), ("method",)))
        await pilot.press("enter")
        seen["by method"] = top_groups(app)
        await pilot.press("enter", *moves(("method",), ("dataset", "method")))
        await pilot.press("enter")
        seen["by dataset, method"] = top_groups(app)
        tree = app.query_one(Tree)
        seen["below cube"] = [
            str(node.label) for node in tree.root.children[0].children
        ]
        await pilot.press("q")

    browse(
        planning_interpreter, scenario, view=PlanView(grouping=("dataset",))
    )

    assert seen["initial"] == ["cube (3 Run, 1 Reuse)"]
    assert seen["by method"] == ["alpha (2 Run, 1 Reuse)", "beta (1 Run)"]
    assert seen["by dataset, method"] == ["cube (3 Run, 1 Reuse)"]
    assert seen["below cube"] == ["alpha (2 Run, 1 Reuse)", "beta (1 Run)"]


def test_filters_compose_and_the_status_line_tells_shown_from_whole(
    planning_interpreter: list[str],
) -> None:
    seen: dict[str, object] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        # The stage filter: its second option, training.
        await pilot.press("f", "enter", "down", "enter")
        seen["training"] = top_groups(app)
        seen["training status"] = status(app)
        # Three controls on, the status filter: its third option, Reuse.
        await pilot.press("right", "right", "right", "enter", "down", "down")
        await pilot.press("enter")
        seen["reused training"] = top_groups(app)
        await pilot.press("q")

    browse(planning_interpreter, scenario)

    assert seen["training"] == ["training (2 Run, 1 Reuse)"]
    assert seen["training status"] == (
        "Shown: 3 of 4 jobs: 2 Run, 1 Reuse | Whole plan: 4 jobs: 3 Run, "
        "1 Reuse"
    )
    assert seen["reused training"] == ["training (1 Reuse)"]


def test_search_narrows_the_tree_and_says_when_nothing_matches(
    planning_interpreter: list[str],
) -> None:
    seen: dict[str, object] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        await pilot.press("slash", *"beta")
        seen["beta"] = top_groups(app)
        # The matches are expanded down to the job.
        seen["job"] = [
            str(node.label)
            for node in app.query_one(Tree)
            .root.children[0]
            .children[0]
            .children[0]
            .children
        ]
        await pilot.press("q", "space", "f", "i", "t")
        seen["nothing"] = top_groups(app)
        seen["nothing status"] = status(app)
        await pilot.press("escape")
        seen["focused"] = focused(app)
        await pilot.press("q")

    app = browse(planning_interpreter, scenario)

    assert seen["beta"] == ["training (1 Run)"]
    assert seen["job"] == [
        "Run train_method: realization 0, train hard budget 4"
    ]
    assert seen["nothing"] == ["No job matches the filters and search"]
    assert str(seen["nothing status"]).startswith(
        "Shown: no job matches the filters and search"
    )
    assert seen["focused"] == "tree"
    assert app.return_code == 0


def investigated(
    rule: str,
    jobid: int,
    identity: JobIdentity | None,
    status: JobStatus = "run",
    dependencies: frozenset[int] = frozenset(),
) -> PlannedJob:
    return PlannedJob(
        rule=rule,
        wildcards={},
        identity=identity,
        device="cpu",
        cpus=1,
        gpus=0,
        jobid=jobid,
        status=status,
        reasons=(
            StatusReason(kind="outputs_up_to_date")
            if status == "reuse"
            else StatusReason(kind="missing_output", files=(f"{rule}.out",)),
        ),
        dependencies=dependencies,
    )


def trained(method: str) -> JobIdentity:
    return JobIdentity(
        stage="training",
        name=method,
        dataset_key="cube",
        dataset_realization_index=0,
        train_hard_budget=4,
    )


# What drives alpha's training: a dataset and a classifier, both shared
# prerequisites of alpha and beta; alpha's evaluation and an aggregation
# job without a job record follow it.
DATASET_GENERATION = investigated(
    "dataset_generation",
    10,
    JobIdentity(
        stage="dataset_generation",
        name="cube",
        dataset_key="cube",
        dataset_realization_index=0,
    ),
    status="reuse",
)
CLASSIFIER_TRAINING = replace(
    DATASET_GENERATION,
    rule="train_classifier",
    identity=JobIdentity(
        stage="classifier_training",
        name="masked_mlp_classifier",
        dataset_key="cube",
        dataset_realization_index=0,
    ),
    placement={"slurm_partition": "cpu-queue"},
    jobid=20,
    dependencies=frozenset({10}),
)
# Revealing a command must not run it.
RAN_MARKER = "preview-ran-this"
ALPHA_TRAINING = replace(
    investigated("train_method", 1, trained("alpha")),
    device="cuda",
    gpus=1,
    shell_command=f"touch {RAN_MARKER}",
    dependencies=frozenset({10, 20}),
)
ALPHA_EVALUATION = investigated(
    "eval_method",
    4,
    replace(trained("alpha"), stage="evaluation", eval_hard_budget=2),
    dependencies=frozenset({1, 20}),
)
MERGE = investigated("merge_eval_perf", 30, None, dependencies=frozenset({4}))
INVESTIGATED = PipelinePlan(
    jobs=[
        DATASET_GENERATION,
        CLASSIFIER_TRAINING,
        ALPHA_TRAINING,
        investigated(
            "train_method",
            3,
            trained("beta"),
            dependencies=frozenset({10, 20}),
        ),
        ALPHA_EVALUATION,
        MERGE,
    ]
)
ALPHA_TRAINING_LABEL = "Run train_method: realization 0, train hard budget 4"
ALPHA_EVALUATION_LABEL = (
    "Run eval_method: realization 0, train hard budget 4, eval hard budget 2"
)


async def open_job(pilot: Pilot[None], groups_before: int) -> None:
    """Expand a top group, its first subgroups, and go to its first job."""
    await pilot.press("down", *["down"] * groups_before)
    await pilot.press("right", "down", "right", "down", "right", "down")


def relations(app: PipelinePreviewApp) -> list[str]:
    options = app.query_one("#relations", OptionList)
    return [
        str(options.get_option_at_index(index).prompt)
        for index in range(options.option_count)
    ]


def highlighted_relation(app: PipelinePreviewApp) -> int | None:
    return app.query_one("#relations", OptionList).highlighted


@pytest.mark.parametrize("plan", [INVESTIGATED])
def test_enter_follows_a_relation_and_backspace_returns(
    planning_interpreter: list[str],
) -> None:
    seen: dict[str, object] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        await open_job(pilot, 2)
        seen["job"] = cursor_label(app)
        await pilot.press("enter", "tab")
        seen["focused"] = focused(app)
        seen["relations"] = relations(app)
        # The third relation: alpha's evaluation, which needs the training.
        await pilot.press("down", "down", "enter")
        await pilot.pause()
        seen["followed"] = cursor_label(app)
        seen["followed details"] = details(app)
        seen["followed relations"] = relations(app)
        seen["focused after following"] = focused(app)
        await pilot.press("backspace")
        await pilot.pause()
        seen["returned"] = cursor_label(app)
        seen["returned details"] = details(app)
        seen["returned relation"] = highlighted_relation(app)
        seen["focused after returning"] = focused(app)
        await pilot.press("q")

    browse(planning_interpreter, scenario)

    assert seen["job"] == ALPHA_TRAINING_LABEL
    assert seen["focused"] == "relations"
    assert seen["relations"] == [
        "Needs Reuse dataset_generation: realization 0 (cube)",
        "Needs Reuse train_classifier: realization 0 "
        "(masked_mlp_classifier, cube)",
        "Needed by Run eval_method: realization 0, train hard budget 4, "
        "eval hard budget 2 (alpha, cube)",
    ]
    # The tree follows to the evaluation, so it can be browsed from there.
    assert seen["followed"] == ALPHA_EVALUATION_LABEL
    assert "Training hard budget: 4" in str(seen["followed details"])
    assert "Evaluation hard budget: 2" in str(seen["followed details"])
    assert seen["followed relations"] == [
        "Needs Reuse train_classifier: realization 0 "
        "(masked_mlp_classifier, cube)",
        "Needs Run train_method: realization 0, train hard budget 4 "
        "(alpha, cube)",
        "Needed by Run merge_eval_perf",
    ]
    assert seen["focused after following"] == "relations"
    assert seen["returned"] == ALPHA_TRAINING_LABEL
    assert "Device: cuda" in str(seen["returned details"])
    assert seen["returned relation"] == 2
    assert seen["focused after returning"] == "relations"


@pytest.mark.parametrize("plan", [INVESTIGATED])
def test_a_dependency_the_filters_hide_is_inspected_without_changing_them(
    planning_interpreter: list[str],
) -> None:
    seen: dict[str, object] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        seen["status"] = status(app)
        await open_job(pilot, 0)
        await pilot.press("enter", "tab")
        seen["relations"] = relations(app)
        # The classifier, which the evaluation filter hides
        await pilot.press("enter")
        await pilot.pause()
        seen["followed details"] = details(app)
        seen["followed relations"] = relations(app)
        seen["status after following"] = status(app)
        seen["tree after following"] = top_groups(app)
        seen["cursor after following"] = cursor_label(app)
        # Enter in the tree shows its job's details again.
        await pilot.press("escape", "enter")
        seen["tree job details"] = details(app)
        await pilot.press("backspace")
        await pilot.pause()
        seen["returned details"] = details(app)
        await pilot.press("q")

    browse(
        planning_interpreter,
        scenario,
        view=PlanView(stage="evaluation"),
    )

    outside = " — outside the filtered tree"
    assert seen["relations"] == [
        "Needs Reuse train_classifier: realization 0 "
        f"(masked_mlp_classifier, cube){outside}",
        "Needs Run train_method: realization 0, train hard budget 4 "
        f"(alpha, cube){outside}",
        f"Needed by Run merge_eval_perf{outside}",
    ]
    followed = str(seen["followed details"])
    assert followed.startswith(OUTSIDE_FILTERED_TREE)
    # A shared prerequisite: its classifier is no method; it names the
    # methods consuming it, and where it would run.
    assert "Rule: train_classifier" in followed
    assert "Method: not applicable" in followed
    assert "Classifier: masked_mlp_classifier" in followed
    assert "Consuming methods: alpha, beta" in followed
    assert "Placement: slurm_partition=cpu-queue" in followed
    assert seen["followed relations"] == [
        f"Needs Reuse dataset_generation: realization 0 (cube){outside}",
        "Needed by Run eval_method: realization 0, train hard budget 4, "
        "eval hard budget 2 (alpha, cube)",
        "Needed by Run train_method: realization 0, train hard budget 4 "
        f"(alpha, cube){outside}",
        "Needed by Run train_method: realization 0, train hard budget 4 "
        f"(beta, cube){outside}",
    ]
    # The filters, the tree and its counts stay as they were.
    assert seen["status after following"] == seen["status"]
    assert seen["status"] == (
        "Shown: 1 of 6 jobs: 1 Run | Whole plan: 6 jobs: 4 Run, 2 Reuse"
    )
    assert seen["tree after following"] == ["evaluation (1 Run)"]
    assert seen["cursor after following"] == ALPHA_EVALUATION_LABEL
    assert "Rule: eval_method" in str(seen["tree job details"])
    returned = str(seen["returned details"])
    assert not returned.startswith(OUTSIDE_FILTERED_TREE)
    assert "Rule: eval_method" in returned


@pytest.mark.parametrize("plan", [INVESTIGATED])
def test_a_jobs_shell_command_is_revealed_apart_and_never_run(
    plannings: Path, planning_interpreter: list[str]
) -> None:
    seen: dict[str, object] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        await planned(pilot, app)
        await open_job(pilot, 2)
        seen["details"] = details(app)
        await pilot.press("s")
        seen["command"] = app.screen.query_one(TextArea).text
        await pilot.press("c")
        seen["clipboard"] = app.clipboard
        await pilot.press("escape")
        seen["after closing"] = cursor_label(app)
        # Beta, evaluation, then the aggregation job without a job record
        await pilot.press("down", "down", "down")
        await pilot.press("right", "down", "right", "down", "right", "down")
        seen["record-less details"] = details(app)
        await pilot.press("s")
        seen["no command"] = app.screen.query_one(TextArea).text
        await pilot.press("escape", "q")

    app = browse(planning_interpreter, scenario)

    assert seen["command"] == f"touch {RAN_MARKER}"
    assert seen["clipboard"] == seen["command"]
    assert seen["command"] not in str(seen["details"])
    assert seen["after closing"] == ALPHA_TRAINING_LABEL
    record_less = str(seen["record-less details"])
    assert "Job record: none: the rule writes no job record" in record_less
    assert "Method: not applicable" in record_less
    assert str(seen["no command"]).startswith("No shell command")
    # Browsing ran nothing and planned once.
    assert not Path(RAN_MARKER).exists()
    assert plannings.read_text().count("\n") == 1
    assert app.return_code == 0


@pytest.mark.parametrize("plan", [INVESTIGATED])
def test_relations_are_followed_in_a_narrow_terminal_too(
    planning_interpreter: list[str],
) -> None:
    shown: list[tuple[bool, bool]] = []
    seen: dict[str, object] = {}

    async def scenario(pilot: Pilot[None], app: PipelinePreviewApp) -> None:
        tree = app.query_one("#tree")
        pane = app.query_one("#details-pane")
        await planned(pilot, app)
        await open_job(pilot, 2)
        await pilot.press("enter", "tab", "enter")
        await pilot.pause()
        shown.append((tree.display, pane.display))
        seen["followed"] = relations(app)
        await pilot.press("backspace")
        await pilot.pause()
        shown.append((tree.display, pane.display))
        await pilot.press("escape")
        shown.append((tree.display, pane.display))
        seen["job"] = cursor_label(app)
        # Back to the job's group, still in the tree
        await pilot.press("left")
        seen["group"] = cursor_label(app)
        seen["focused"] = focused(app)
        seen["group details"] = details(app)
        seen["relations shown"] = app.query_one("#relations").display
        await pilot.press("q")

    browse(planning_interpreter, scenario, size=NARROW)

    assert shown == [(False, True), (False, True), (True, False)]
    assert seen["followed"] == [
        "Needed by Reuse train_classifier: realization 0 "
        "(masked_mlp_classifier, cube)",
        "Needed by Run train_method: realization 0, train hard budget 4 "
        "(alpha, cube)",
        "Needed by Run train_method: realization 0, train hard budget 4 "
        "(beta, cube)",
    ]
    assert seen["job"] == ALPHA_TRAINING_LABEL
    assert seen["group"] == "cube (1 Run)"
    assert seen["focused"] == "tree"
    assert str(seen["group details"]).startswith("Dataset: cube")
    assert seen["relations shown"] is False


def alive(pid: int) -> bool:
    """Whether the process exists; a reaped child does not."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True
