"""The preview's entry point: its grouping, and refusing to open without a terminal."""

import pytest

from afabench.pipeline_preview.__main__ import main, preview_arguments


def test_without_a_terminal_the_preview_points_to_a_dry_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The test runner's streams are not a terminal.
    assert main(["--profile", "workflow/profiles/pipeline/all", "all"]) == 2
    error = capsys.readouterr().err
    assert "terminal" in error
    assert "-n" in error


def test_a_leading_group_by_sets_the_initial_grouping() -> None:
    assert preview_arguments(
        ["--group-by", "dataset,method", "--profile", "p", "all"]
    ) == (("dataset", "method"), ["--profile", "p", "all"])
    assert preview_arguments(["--profile", "p", "all"]) == (
        ("stage", "method", "dataset"),
        ["--profile", "p", "all"],
    )


def test_an_invalid_grouping_is_refused_before_anything_opens(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["--group-by", "method,method", "--profile", "p", "all"]) == 2
    assert "'method' repeats" in capsys.readouterr().err
