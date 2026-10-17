"""
Open the interactive preview of a pipeline invocation.

    python -m afabench.pipeline_preview <Snakemake arguments>

The pipeline launcher runs this for `--interactive-preview`, with the
Snakemake arguments it resolved for the run, in the environment it
selected. Needs a terminal on stdin and stdout; exits with 1 when planning
failed, 2 without a terminal, else 0.
"""

import sys
from collections.abc import Sequence

from afabench.pipeline_preview.app import PipelinePreviewApp
from afabench.pipeline_preview.planning_process import PlanningProcess

NO_TERMINAL_EXIT_STATUS = 2


def main(snakemake_arguments: Sequence[str]) -> int:
    """Browse the plan of `snakemake <snakemake_arguments>`; return the exit status."""
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "The interactive preview needs an interactive terminal on stdin "
            "and stdout; for a preview in plain text, give the pipeline "
            "launcher -n instead",
            file=sys.stderr,
        )
        return NO_TERMINAL_EXIT_STATUS
    app = PipelinePreviewApp(PlanningProcess(snakemake_arguments))
    app.run()
    return app.return_code or 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
