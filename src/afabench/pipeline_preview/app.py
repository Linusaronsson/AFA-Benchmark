"""
The terminal application browsing a pipeline plan (`--interactive-preview`).

It opens at once with a planning indicator while a child process plans the
invocation (`planning_process`), then shows the plan's jobs, run and
reused, as a tree, stage → method → dataset → job, beside the selected
group's summary or job's details (`model`). Each job wears its status as a
colored text badge, and each group its counts by status. A release
preview with conditionally reused jobs keeps a warning that execution
checks the regenerated datasets they wait for. A narrow terminal
shows one pane at a time. When planning fails, the child's diagnostics
replace the panes and quitting exits with 1. Browsing changes nothing: the
app runs no job and writes no file.
"""

import asyncio
from typing import TYPE_CHECKING, ClassVar, Literal, final, override

from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, Static, TextArea, Tree
from textual.widgets.tree import TreeNode

from afabench.pipeline_plan.planning import JobStatus
from afabench.pipeline_preview.model import (
    STATUS_LABELS,
    GroupNode,
    JobNode,
    group_summary,
    job_details,
    plan_summary,
    preview_tree,
    release_warning,
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
            self.select_node(node.parent)


@final
class PipelinePreviewApp(App[None]):
    """Browse the jobs a pipeline invocation would run and reuse."""

    TITLE = "Pipeline preview"
    SUB_TITLE = "jobs run and reused"
    CSS = """
    #status { padding: 0 1; height: 1; }
    #release-warning { padding: 0 1; height: auto; background: $warning; color: $text; text-style: bold; }
    #panes { height: 1fr; }
    #tree { width: 1fr; border: round $primary; }
    #details { width: 1fr; border: round $secondary; padding: 0 1; }
    #diagnostics { height: 1fr; border: round $error; }
    """
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        # Over the screen's own tab binding, which skips a hidden pane.
        Binding("tab", "switch_pane", "Switch pane", priority=True),
        Binding("enter", "show_details", "Details"),
        Binding("escape", "show_tree", "Tree", show=False),
        Binding("c", "copy_diagnostics", "Copy diagnostics"),
    ]

    def __init__(self, planning: PlanningProcess) -> None:
        super().__init__()
        self.planning: PlanningProcess = planning
        self.planning_state: PlanningState = "planning"
        self.plan: PipelinePlan | None = None
        # The child's output, shown when planning failed
        self.diagnostics: str = ""
        self.narrow: bool = False

    @override
    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Planning the invocation…", id="status")
        warning = Static("", id="release-warning")
        warning.display = False
        yield warning
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
            for pane in self.query("#tree, #details"):
                pane.display = False
            diagnostics = self.query_one(TextArea)
            diagnostics.load_text(result.diagnostics)
            diagnostics.display = True
            diagnostics.focus()
            return
        self.planning_state = "planned"
        self.plan = result.plan
        self.query_one("#status", Static).update(plan_summary(result.plan))
        warning = release_warning(result.plan)
        if warning is not None:
            banner = self.query_one("#release-warning", Static)
            banner.update(warning)
            banner.display = True
        tree = self.query_one(PlanTree)
        for group in preview_tree(result.plan):
            self._add_group(tree.root, group)

    def _add_group(
        self, parent: TreeNode[PreviewNode], group: GroupNode
    ) -> None:
        node = parent.add(f"{group.label} ({group.counts_label})", data=group)
        for child in group.children:
            if isinstance(child, GroupNode):
                self._add_group(node, child)
            else:
                node.add_leaf(_badged(child), data=child)

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
    return Text.assemble(
        (STATUS_LABELS[job.status], STATUS_STYLES[job.status]),
        " ",
        job.label,
    )
