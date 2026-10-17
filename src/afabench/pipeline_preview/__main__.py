"""
Open the interactive preview of a pipeline invocation.

    python -m afabench.pipeline_preview [--group-by <dimensions>] \
        -- <Snakemake arguments>

The pipeline launcher runs this for `--interactive-preview`, with the
Snakemake arguments it resolved for the run, in the environment it
selected. `--group-by` is the initial grouping: comma-separated dimensions
among stage, method and dataset, each once; by default
stage,method,dataset. Needs a terminal on stdin and stdout; exits with 1
when planning failed, 2 for an invalid grouping or without a terminal,
else 0.
"""

import sys
from typing import Annotated

import typer

from afabench.pipeline_preview.app import PipelinePreviewApp
from afabench.pipeline_preview.model import (
    DEFAULT_GROUPING,
    Dimension,
    GroupingError,
    PlanView,
    parse_grouping,
)
from afabench.pipeline_preview.planning_process import PlanningProcess

USAGE_EXIT_STATUS = 2

app = typer.Typer()


def in_terminal() -> bool:
    """Whether stdin and stdout are an interactive terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def _grouping(text: str | None) -> tuple[Dimension, ...]:
    if text is None:
        return DEFAULT_GROUPING
    try:
        return parse_grouping(text)
    except GroupingError as error:
        raise typer.BadParameter(str(error)) from error


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
def main(
    ctx: typer.Context,
    group_by: Annotated[
        str | None,
        typer.Option(
            metavar="DIMENSIONS",
            help="The initial grouping: stage, method and dataset, or some "
            "of them, each once, comma-separated in order.",
        ),
    ] = None,
) -> None:
    """Browse the plan of `snakemake <arguments>`, given after --."""
    grouping = _grouping(group_by)
    if not in_terminal():
        typer.echo(
            "The interactive preview needs an interactive terminal on stdin "
            "and stdout; for a preview in plain text, give the pipeline "
            "launcher -n instead",
            err=True,
        )
        raise typer.Exit(USAGE_EXIT_STATUS)
    preview = PipelinePreviewApp(
        PlanningProcess(ctx.args), view=PlanView(grouping=grouping)
    )
    preview.run()
    raise typer.Exit(preview.return_code or 0)


if __name__ == "__main__":
    app()
