"""
Open the interactive preview of a pipeline invocation.

    python -m afabench.pipeline_preview [--group-by <dimensions>] \
        <Snakemake arguments>

The pipeline launcher runs this for `--interactive-preview`, with the
Snakemake arguments it resolved for the run, in the environment it
selected. `--group-by`, first if given, is the initial grouping:
comma-separated dimensions among stage, method and dataset, each once;
by default stage,method,dataset. Needs a terminal on stdin and stdout;
exits with 1 when planning failed, 2 for an invalid grouping or without a
terminal, else 0.
"""

import sys
from collections.abc import Sequence

from afabench.pipeline_preview.app import PipelinePreviewApp
from afabench.pipeline_preview.model import (
    GROUPING,
    Dimension,
    GroupingError,
    PlanView,
    parse_grouping,
)
from afabench.pipeline_preview.planning_process import PlanningProcess

USAGE_EXIT_STATUS = 2


def preview_arguments(
    arguments: Sequence[str],
) -> tuple[tuple[Dimension, ...], list[str]]:
    """Split the arguments into the initial grouping and Snakemake's."""
    if arguments[:1] == ["--group-by"] and len(arguments) > 1:
        return parse_grouping(arguments[1]), list(arguments[2:])
    return GROUPING, list(arguments)


def main(arguments: Sequence[str]) -> int:
    """Browse the plan of the Snakemake arguments; return the exit status."""
    try:
        grouping, snakemake_arguments = preview_arguments(arguments)
    except GroupingError as error:
        print(f"--group-by: {error}", file=sys.stderr)
        return USAGE_EXIT_STATUS
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "The interactive preview needs an interactive terminal on stdin "
            "and stdout; for a preview in plain text, give the pipeline "
            "launcher -n instead",
            file=sys.stderr,
        )
        return USAGE_EXIT_STATUS
    app = PipelinePreviewApp(
        PlanningProcess(snakemake_arguments),
        view=PlanView(grouping=grouping),
    )
    app.run()
    return app.return_code or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
