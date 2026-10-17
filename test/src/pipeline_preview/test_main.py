"""The preview's entry point refuses to open without a terminal (#158)."""

import pytest

from afabench.pipeline_preview.__main__ import main


def test_without_a_terminal_the_preview_points_to_a_dry_run(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The test runner's streams are not a terminal.
    assert main(["--profile", "workflow/profiles/pipeline/all", "all"]) == 2
    error = capsys.readouterr().err
    assert "terminal" in error
    assert "-n" in error
