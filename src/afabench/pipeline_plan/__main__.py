"""
Plan a pipeline invocation in its own process and write the plan as JSON.

    python -m afabench.pipeline_plan --output <plan.json> -- <Snakemake arguments>

The Snakemake arguments are the launcher's resolved ones
(`workflow.run.snakemake_arguments`). Snakemake's diagnostics go to the
usual streams; an invocation Snakemake rejects exits with 1 and writes no
plan. The interactive preview runs this as its planning child process.
"""

from pathlib import Path
from typing import Annotated

import typer

from afabench.pipeline_plan.plan_json import dump_plan
from afabench.pipeline_plan.planning import InvocationError, plan_invocation

app = typer.Typer()


@app.command(
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    }
)
def main(
    ctx: typer.Context,
    output: Annotated[Path, typer.Option(help="Write the plan here as JSON.")],
) -> None:
    """Plan `snakemake <arguments>` and write the plan to --output."""
    try:
        plan = plan_invocation(ctx.args)
    except InvocationError:
        raise typer.Exit(1) from None
    output.write_text(dump_plan(plan))


if __name__ == "__main__":
    app()
