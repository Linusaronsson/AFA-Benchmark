"""The preview's entry point: its grouping, and refusing to open without a terminal."""

from collections.abc import Sequence

import pytest
from typer.testing import CliRunner

from afabench.pipeline_preview import __main__ as entry_point
from afabench.pipeline_preview.model import PlanView
from afabench.pipeline_preview.planning_process import PlanningProcess

ARGUMENTS = ["--profile", "workflow/profiles/pipeline/all", "all"]


def invoke(
    arguments: Sequence[str],
    monkeypatch: pytest.MonkeyPatch,
    *,
    terminal: bool = True,
) -> tuple[int, str, list[tuple[list[str], PlanView]]]:
    """Run the entry point; return its exit status, stderr and what it opened."""
    opened: list[tuple[list[str], PlanView]] = []

    class OpenedPreview:
        """Stands in for the app: records what it would browse."""

        def __init__(
            self, planning: PlanningProcess, *, view: PlanView
        ) -> None:
            opened.append((planning.snakemake_arguments, view))
            # As after a planning failure
            self.return_code = 1

        def run(self) -> None:
            pass

    monkeypatch.setattr(entry_point, "PipelinePreviewApp", OpenedPreview)
    monkeypatch.setattr(entry_point, "in_terminal", lambda: terminal)
    result = CliRunner().invoke(entry_point.app, list(arguments))
    return result.exit_code, result.stderr, opened


def test_the_preview_browses_the_snakemake_arguments_after_the_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, _, opened = invoke(["--", *ARGUMENTS, "-n"], monkeypatch)

    assert opened == [([*ARGUMENTS, "-n"], PlanView())]
    assert status == 1


def test_group_by_sets_the_initial_grouping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, opened = invoke(
        ["--group-by", "dataset,method", "--", *ARGUMENTS], monkeypatch
    )

    assert opened == [
        (ARGUMENTS, PlanView(grouping=("dataset", "method"))),
    ]


def test_without_a_terminal_the_preview_points_to_a_dry_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, error, opened = invoke(
        ["--", *ARGUMENTS], monkeypatch, terminal=False
    )

    assert status == 2
    assert "terminal" in error
    assert "-n" in error
    assert opened == []


def test_an_invalid_grouping_is_refused_before_anything_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status, error, opened = invoke(
        ["--group-by", "method,method", "--", *ARGUMENTS], monkeypatch
    )

    assert status == 2
    assert "'method' repeats" in error
    assert opened == []
