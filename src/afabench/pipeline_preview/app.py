"""
The terminal application browsing a pipeline plan (`--interactive-preview`).

It opens at once with a planning indicator while a child process plans the
invocation (`planning_process`), then shows the plan's jobs, run and
reused, as a tree, stage → method → dataset → job by default, beside the
selected group's summary or job's details (`model`). Each job wears its
status as a colored text badge, and each group its counts by status.
Controls above the tree regroup the jobs, filter them by stage, method,
dataset and status, and search them; all of it reshapes the loaded plan,
and nothing plans again. Below a job's details, its relations, the jobs
it needs and the jobs needing it, can be followed, also to jobs the
filters hide, and returned from; its shell command shows apart, on
request. A release preview with conditionally reused jobs
keeps a warning that execution checks the regenerated datasets they wait
for. A narrow terminal shows one pane at a time. When planning fails, the
child's diagnostics replace the panes and quitting exits with 1. Browsing
changes nothing: the app runs no job and writes no file.
"""

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Literal, final, override

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    Select,
    Static,
    TextArea,
    Tree,
)
from textual.widgets.option_list import Option
from textual.widgets.tree import TreeNode

from afabench.pipeline_plan.planning import JobStatus, PlannedJob
from afabench.pipeline_preview.model import (
    DEFAULT_VIEW,
    DIMENSIONS,
    GROUPINGS,
    OUTSIDE_FILTERED_TREE,
    STATUS_LABELS,
    GroupNode,
    JobNode,
    PlanView,
    Relation,
    filter_values,
    group_summary,
    job_details,
    job_relations,
    job_title,
    plan_statuses,
    preview_tree,
    release_warning,
    shell_command_text,
    shown_jobs,
    view_summary,
)
from afabench.pipeline_preview.planning_process import (
    PlanningProcess,
    PlanningResult,
)

if TYPE_CHECKING:
    from afabench.pipeline_plan.planning import PipelinePlan

type PlanningState = Literal["planning", "planned", "failed"]
type PreviewNode = GroupNode | JobNode
# Below this many columns, the tree and the details take turns.
NARROW_WIDTH = 80
# The view's fields the filter controls set, in the controls' order
FILTERS = (*DIMENSIONS, "status")
# What each filter control shows when it filters nothing
FILTER_PROMPTS = {
    "stage": "All stages",
    "method": "All methods",
    "dataset": "All datasets",
    "status": "All statuses",
}
NO_MATCH = "No job matches the filters and search"
NO_RELATIONS = "Needs no job of this plan, and no job of it needs this one"
DIRECTION_LABELS = {"needs": "Needs", "needed_by": "Needed by"}
# The color of each status's badge, beside its text
STATUS_STYLES: dict[JobStatus, str] = {
    "run": "bold white on dark_green",
    "reuse": "bold white on dark_blue",
    "conditional_reuse": "bold black on dark_orange",
}


@final
class PlanTree(Tree[PreviewNode]):
    """The grouped jobs; the arrow keys walk, expand and collapse them."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("right", "expand_cursor", "Expand", show=False),
        Binding("left", "collapse_cursor", "Collapse", show=False),
    ]

    def action_expand_cursor(self) -> None:
        node = self.cursor_node
        if node is not None and node.allow_expand:
            node.expand()

    def action_collapse_cursor(self) -> None:
        node = self.cursor_node
        if node is None:
            return
        if node.is_expanded:
            node.collapse()
        elif node.parent is not None and node.parent is not self.root:
            # Moving, not selecting, keeps the focus in the tree.
            self.move_cursor(node.parent)


@final
class Controls(Container):
    """The grouping and filters; the arrow keys move between them."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("left", "move(-1)", "Previous control", show=False),
        Binding("right", "move(1)", "Next control", show=False),
    ]

    def action_move(self, step: int) -> None:
        selects = list(self.query(Select))
        for index, select in enumerate(selects):
            if select is self.app.focused:
                selects[(index + step) % len(selects)].focus()
                return


@final
class ShellCommandScreen(ModalScreen[None]):
    """A job's shell command, or why it has none, over the browser."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "dismiss", "Close"),
        Binding("c", "copy", "Copy"),
    ]
    CSS = """
    ShellCommandScreen { align: center middle; }
    #command { width: 90%; height: 70%; border: round $accent; }
    """

    def __init__(self, title: str, command: str) -> None:
        super().__init__()
        self._title: str = title
        self._command: str = command

    @override
    def compose(self) -> ComposeResult:
        command = TextArea(
            self._command, id="command", read_only=True, soft_wrap=True
        )
        command.border_title = f"Shell command of {self._title}"
        command.border_subtitle = "Escape closes, c copies; nothing runs"
        yield command
        yield Footer()

    def action_copy(self) -> None:
        self.app.copy_to_clipboard(self._command)


@final
class PipelinePreviewApp(App[None]):
    """Browse the jobs a pipeline invocation would run and reuse."""

    TITLE = "Pipeline preview"
    SUB_TITLE = "jobs run and reused"
    CSS = """
    #controls {
        layout: grid;
        grid-size: 5;
        grid-columns: 2fr 1fr 1fr 1fr 1fr;
        grid-rows: 1;
        height: auto;
    }
    #controls.-narrow { grid-size: 2; grid-columns: 1fr 1fr; }
    #grouping-control { height: 1; }
    #grouping-control Label { padding: 0 1; }
    #search { height: 1; }
    #status { padding: 0 1; height: 1; }
    #release-warning { padding: 0 1; height: auto; background: $warning; color: $text; text-style: bold; }
    #panes { height: 1fr; }
    #tree { width: 1fr; border: round $primary; }
    #details-pane { width: 1fr; }
    #details { height: 1fr; border: round $secondary; padding: 0 1; }
    #relations { height: auto; max-height: 50%; border: round $secondary; }
    #diagnostics { height: 1fr; border: round $error; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("g", "focus_grouping", "Group by"),
        Binding("f", "focus_filters", "Filter"),
        Binding("slash", "search", "Search"),
        # Over the screen's own tab binding, which skips a hidden pane.
        Binding("tab", "switch_pane", "Switch pane", priority=True),
        Binding("enter", "show_details", "Details"),
        Binding("escape", "show_tree", "Tree", show=False),
        Binding("backspace", "go_back", "Back"),
        Binding("s", "show_command", "Shell command"),
        Binding("c", "copy_diagnostics", "Copy diagnostics"),
    ]

    def __init__(
        self, planning: PlanningProcess, *, view: PlanView = DEFAULT_VIEW
    ) -> None:
        super().__init__()
        self.planning: PlanningProcess = planning
        # The grouping, filters and search the tree shows the plan in
        self.view: PlanView = view
        self.planning_state: PlanningState = "planning"
        self.plan: PipelinePlan | None = None
        # The child's output, shown when planning failed
        self.diagnostics: str = ""
        self.narrow: bool = False
        # The job whose details show, None for a group or nothing
        self.inspected: PlannedJob | None = None
        # The relations listed below the inspected job's details
        self._relations: list[Relation] = []
        # The jobs inspected before following a relation, each with the
        # relation followed from it, latest last
        self._followed_from: list[tuple[PlannedJob, int]] = []

    @override
    def compose(self) -> ComposeResult:
        yield Header()
        with Controls(id="controls", disabled=True):
            with Horizontal(id="grouping-control"):
                yield Label("Group by")
                yield Select(
                    [
                        (" → ".join(grouping), grouping)
                        for grouping in GROUPINGS
                    ],
                    value=self.view.grouping,
                    allow_blank=False,
                    compact=True,
                    id="grouping",
                )
            for name in FILTERS:
                yield Select[str](
                    [],
                    prompt=FILTER_PROMPTS[name],
                    compact=True,
                    id=f"filter-{name}",
                )
        yield Input(
            placeholder="/ searches rules, stages, methods, datasets and job labels",
            compact=True,
            disabled=True,
            id="search",
        )
        yield Static("Planning the invocation…", id="status")
        warning = Static("", id="release-warning")
        warning.display = False
        yield warning
        with Horizontal(id="panes"):
            tree = PlanTree("Jobs", id="tree")
            tree.show_root = False
            tree.border_title = "Jobs (↑↓ move, → expand, ← collapse)"
            yield tree
            with Vertical(id="details-pane"):
                with VerticalScroll(id="details", can_focus=True) as details:
                    details.border_title = "Details"
                    yield Static("", id="details-text")
                relations = OptionList(id="relations")
                relations.border_title = (
                    "Needs and needed by (Enter follows, Backspace goes back)"
                )
                relations.display = False
                yield relations
        diagnostics = TextArea(id="diagnostics", read_only=True)
        diagnostics.display = False
        diagnostics.border_title = "Planning diagnostics (select, c copies)"
        yield diagnostics
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(PlanTree).focus()
        self.planning.start()
        self.plan_in_child()

    @work(exclusive=True)
    async def plan_in_child(self) -> None:
        # Blocking in a thread keeps the terminal responsive; quitting
        # cancels the child, which ends the wait.
        result = await asyncio.to_thread(self.planning.wait)
        self.show_result(result)

    def show_result(self, result: PlanningResult) -> None:
        self.diagnostics = result.diagnostics
        if result.plan is None:
            self.planning_state = "failed"
            self.query_one("#status", Static).update(
                "Planning failed: the invocation was not planned; q quits with exit status 1"
            )
            for pane in self.query("#controls, #search, #tree, #details-pane"):
                pane.display = False
            diagnostics = self.query_one(TextArea)
            diagnostics.load_text(result.diagnostics)
            diagnostics.display = True
            diagnostics.focus()
            return
        self.planning_state = "planned"
        self.plan = result.plan
        warning = release_warning(result.plan)
        if warning is not None:
            banner = self.query_one("#release-warning", Static)
            banner.update(warning)
            banner.display = True
        for dimension in DIMENSIONS:
            self.query_one(f"#filter-{dimension}", Select).set_options(
                (value, value)
                for value in filter_values(result.plan, dimension)
            )
        self.query_one("#filter-status", Select).set_options(
            (_badge(status), status) for status in plan_statuses(result.plan)
        )
        self.query_one(Controls).disabled = False
        self.query_one("#search", Input).disabled = False
        self._show_view()

    def _show_view(self) -> None:
        """Show the plan's jobs as the view groups, filters and searches them."""
        assert self.plan is not None
        self.query_one("#status", Static).update(
            view_summary(self.plan, self.view)
        )
        # The details go with the old tree.
        self._show_group(None)
        tree = self.query_one(PlanTree)
        tree.clear()
        groups = preview_tree(self.plan, self.view)
        for group in groups:
            self._add_group(tree.root, group)
        if not groups:
            tree.root.add_leaf(NO_MATCH)
        elif self.view.search:
            tree.root.expand_all()

    def _add_group(
        self, parent: TreeNode[PreviewNode], group: GroupNode
    ) -> None:
        node = parent.add(f"{group.label} ({group.counts_label})", data=group)
        for child in group.children:
            if isinstance(child, GroupNode):
                self._add_group(node, child)
            else:
                node.add_leaf(_badged(child), data=child)

    @on(Select.Changed)
    def _change_view(self, event: Select.Changed) -> None:
        selected = event.select.selection
        if event.select.id == "grouping":
            view = replace(self.view, grouping=selected)
        else:
            name = (event.select.id or "").removeprefix("filter-")
            view = replace(self.view, **{name: selected})
        self._set_view(view)

    @on(Input.Changed, "#search")
    def _search(self, event: Input.Changed) -> None:
        self._set_view(replace(self.view, search=event.value))

    @on(Input.Submitted, "#search")
    def _searched(self) -> None:
        self.action_show_tree()

    def _set_view(self, view: PlanView) -> None:
        if view != self.view:
            self.view = view
            if self.plan is not None:
                self._show_view()

    def action_focus_grouping(self) -> None:
        self._focus_control("#grouping")

    def action_focus_filters(self) -> None:
        self._focus_control(f"#filter-{FILTERS[0]}")

    def action_search(self) -> None:
        self._focus_control("#search")

    def _focus_control(self, selector: str) -> None:
        if self.planning_state == "planned":
            self.screen.set_focus(self.query_one(selector))

    def on_tree_node_highlighted(
        self, event: Tree.NodeHighlighted[PreviewNode]
    ) -> None:
        self._show_node(event.node.data)

    def _show_node(self, node: PreviewNode | None) -> None:
        """Show the tree node's group summary or job details."""
        if not isinstance(node, JobNode):
            self._show_group(node)
        elif self.inspected is None or self.inspected.jobid != node.job.jobid:
            self._inspect(node.job)

    def _show_group(self, group: GroupNode | None) -> None:
        """Show the group's summary, or nothing, in place of job details."""
        self.inspected = None
        self.query_one("#details-text", Static).update(
            "" if group is None else group_summary(group)
        )
        self.query_one("#relations").display = False

    def _inspect(self, job: PlannedJob, relation: int = 0) -> None:
        """Show the job's details and relations, the given one highlighted."""
        assert self.plan is not None
        self.inspected = job
        shown = any(
            shown.jobid == job.jobid
            for shown in shown_jobs(self.plan, self.view)
        )
        self.query_one("#details-text", Static).update(
            "\n".join(
                ([] if shown else [OUTSIDE_FILTERED_TREE, ""])
                + [
                    f"{label}: {value}".replace("\n", "\n    ")
                    for label, value in job_details(job, self.plan)
                ]
            )
        )
        self._relations = job_relations(job, self.plan, self.view)
        relations = self.query_one("#relations", OptionList)
        relations.set_options(
            [Option(_relation_label(related)) for related in self._relations]
            or [Option(NO_RELATIONS, disabled=True)]
        )
        relations.display = True
        if self._relations:
            relations.highlighted = relation

    @on(OptionList.OptionSelected, "#relations")
    def _follow(self, event: OptionList.OptionSelected) -> None:
        """Inspect the selected relation, remembering where it was followed from."""
        assert self.inspected is not None
        self._followed_from.append((self.inspected, event.option_index))
        self._go_to(self._relations[event.option_index].job)

    def action_go_back(self) -> None:
        """Return to the job a relation was last followed from."""
        if self.planning_state != "planned" or not self._followed_from:
            return
        job, relation = self._followed_from.pop()
        self._go_to(job, relation)
        self._focus_pane("#relations")

    def _go_to(self, job: PlannedJob, relation: int = 0) -> None:
        """Inspect the job, with the tree's cursor on it if the tree shows it."""
        self._inspect(job, relation)
        tree = self.query_one(PlanTree)
        node = _tree_node(tree.root, job)
        if node is None:
            return
        parent = node.parent
        while parent is not None:
            parent.expand()
            parent = parent.parent
        # The tree places expanded nodes when it next refreshes.
        self.call_after_refresh(tree.move_cursor, node)

    def action_show_command(self) -> None:
        if self.planning_state != "planned":
            return
        if self.inspected is None:
            self.notify("Select a job to show its shell command")
            return
        self.push_screen(
            ShellCommandScreen(
                job_title(self.inspected), shell_command_text(self.inspected)
            )
        )

    @override
    def check_action(
        self, action: str, parameters: tuple[object, ...]
    ) -> bool | None:
        # Over a shell command, only quitting acts on the browser.
        return action == "quit" or not isinstance(self.screen, ModalScreen)

    def on_tree_node_selected(
        self, event: Tree.NodeSelected[PreviewNode]
    ) -> None:
        event.stop()
        # The details may still show a job a relation was followed to.
        self._show_node(event.node.data)
        self.action_show_details()

    def on_resize(self, event: events.Resize) -> None:
        self.narrow = event.size.width < NARROW_WIDTH
        self._apply_layout()

    def _apply_layout(self) -> None:
        """Side by side, or in a narrow terminal the focused pane only."""
        if self.planning_state == "failed":
            return
        self.query_one(Controls).set_class(self.narrow, "-narrow")
        tree, pane = self.query_one("#tree"), self.query_one("#details-pane")
        # The app's focus moves at once; a widget's has_focus follows later.
        pane_focused = (
            self.focused is not None and pane in self.focused.ancestors
        )
        tree.display = not (self.narrow and pane_focused)
        pane.display = not self.narrow or pane_focused

    def action_switch_pane(self) -> None:
        if self.planning_state == "failed":
            return
        relations = self.query_one("#relations")
        if self.focused is self.query_one("#details") and relations.display:
            self._focus_pane("#relations")
        elif self.focused in (self.query_one("#details"), relations):
            self.action_show_tree()
        else:
            self.action_show_details()

    def action_show_details(self) -> None:
        if self.planning_state != "failed":
            self._focus_pane("#details")

    def action_show_tree(self) -> None:
        if self.planning_state != "failed":
            self._focus_pane("#tree")

    def _focus_pane(self, selector: str) -> None:
        pane = self.query_one(selector)
        # A hidden pane cannot take focus, so show it first. The screen
        # moves the focus at once, where Widget.focus would defer it.
        if selector != "#tree":
            self.query_one("#details-pane").display = True
        pane.display = True
        self.screen.set_focus(pane)
        self._apply_layout()

    def action_copy_diagnostics(self) -> None:
        if self.planning_state == "failed":
            self.copy_to_clipboard(self.diagnostics)

    @override
    async def action_quit(self) -> None:
        self.planning.cancel()
        self.exit(return_code=1 if self.planning_state == "failed" else 0)


def _tree_node(
    parent: TreeNode[PreviewNode], job: PlannedJob
) -> TreeNode[PreviewNode] | None:
    """Return the tree's node of the job, None when the tree omits it."""
    for child in parent.children:
        if (
            isinstance(child.data, JobNode)
            and child.data.job.jobid == job.jobid
        ):
            return child
        found = _tree_node(child, job)
        if found is not None:
            return found
    return None


def _relation_label(relation: Relation) -> Text:
    """Return the related job behind its direction and status badge."""
    return Text.assemble(
        DIRECTION_LABELS[relation.direction],
        " ",
        _badge(relation.job.status),
        " ",
        relation.title,
        ("" if relation.shown else (" — outside the filtered tree", "italic")),
    )


def _badged(job: JobNode) -> Text:
    """Return the job's label behind its status badge, colored by status."""
    return Text.assemble(_badge(job.status), " ", job.label)


def _badge(status: JobStatus) -> Text:
    return Text(STATUS_LABELS[status], STATUS_STYLES[status])
