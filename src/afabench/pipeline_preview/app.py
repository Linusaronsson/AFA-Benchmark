"""
The terminal application browsing a pipeline plan (`--interactive-preview`).

It opens at once with a planning indicator while a child process plans the
invocation (`planning_process`), then shows the plan's jobs, run and
reused, as a tree, stage → method → dataset → job by default, beside the
selected group's summary or job's details (`model`). Each job wears its
status as a colored text badge, and each group its counts by status.
Controls above the tree regroup the jobs, filter them by stage, method,
dataset and status, and search them; all of it reshapes the loaded plan,
and nothing plans again. A narrow terminal shows one pane at a time. When
planning fails, the child's diagnostics replace the panes and quitting
exits with 1. Browsing changes nothing: the app runs no job and writes no
file.
"""

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Literal, final, override

from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import (
    Footer,
    Header,
    Input,
    Label,
    Select,
    Static,
    TextArea,
    Tree,
)
from textual.widgets.tree import TreeNode

from afabench.pipeline_plan.planning import JobStatus
from afabench.pipeline_preview.model import (
    DEFAULT_VIEW,
    DIMENSIONS,
    GROUPINGS,
    STATUS_LABELS,
    GroupNode,
    JobNode,
    PlanView,
    filter_values,
    group_summary,
    job_details,
    plan_statuses,
    preview_tree,
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
NO_MATCH = "No job matches the filters and search"
# The color of each status's badge, beside its text
STATUS_STYLES: dict[JobStatus, str] = {
    "run": "bold white on dark_green",
    "reuse": "bold white on dark_blue",
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
            self.select_node(node.parent)


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
    #panes { height: 1fr; }
    #tree { width: 1fr; border: round $primary; }
    #details { width: 1fr; border: round $secondary; padding: 0 1; }
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
                    prompt=f"All {name}{'es' if name == 'status' else 's'}",
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
        with Horizontal(id="panes"):
            tree = PlanTree("Jobs", id="tree")
            tree.show_root = False
            tree.border_title = "Jobs (↑↓ move, → expand, ← collapse)"
            yield tree
            with VerticalScroll(id="details", can_focus=True) as details:
                details.border_title = "Details"
                yield Static("", id="details-text")
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
            for pane in self.query("#controls, #search, #tree, #details"):
                pane.display = False
            diagnostics = self.query_one(TextArea)
            diagnostics.load_text(result.diagnostics)
            diagnostics.display = True
            diagnostics.focus()
            return
        self.planning_state = "planned"
        self.plan = result.plan
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
        # The details of a job the view may hide go with the old tree.
        self.query_one("#details-text", Static).update("")
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
        self.query_one("#details-text", Static).update(
            self._described(event.node.data)
        )

    def _described(self, node: PreviewNode | None) -> str:
        if node is None:
            return ""
        if isinstance(node, GroupNode):
            return group_summary(node)
        assert self.plan is not None
        return "\n".join(
            f"{label}: {value}".replace("\n", "\n    ")
            for label, value in job_details(node.job, self.plan)
        )

    def on_tree_node_selected(
        self, event: Tree.NodeSelected[PreviewNode]
    ) -> None:
        event.stop()
        self.action_show_details()

    def on_resize(self, event: events.Resize) -> None:
        self.narrow = event.size.width < NARROW_WIDTH
        self._apply_layout()

    def _apply_layout(self) -> None:
        """Side by side, or in a narrow terminal the focused pane only."""
        if self.planning_state == "failed":
            return
        self.query_one(Controls).set_class(self.narrow, "-narrow")
        tree, details = self.query_one("#tree"), self.query_one("#details")
        # The app's focus moves at once; a widget's has_focus follows later.
        details_focused = self.focused is details
        tree.display = not (self.narrow and details_focused)
        details.display = not self.narrow or details_focused

    def action_switch_pane(self) -> None:
        if self.planning_state == "failed":
            return
        if self.focused is self.query_one("#details"):
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


def _badged(job: JobNode) -> Text:
    """Return the job's label behind its status badge, colored by status."""
    return Text.assemble(_badge(job.status), " ", job.label)


def _badge(status: JobStatus) -> Text:
    return Text(STATUS_LABELS[status], STATUS_STYLES[status])
