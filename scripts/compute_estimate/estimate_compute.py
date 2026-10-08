"""
Plan the jobs of a pipeline invocation (`just estimate-compute`).

Takes the same Snakemake arguments as the real run (`--profile`,
`--workflow-profile`, `--config`, `--set-resources`, target, ...) and lists
the jobs that invocation would run, one CSV row per job: its rule, device,
CPUs and GPUs, then one column per wildcard (empty where the job's rule has
no such wildcard). See `afabench.compute_estimate.planning`.

Usage:
    just estimate-compute --output planned_jobs.csv \
        --profile extra/workflow/profiles/config/kdd26 \
        --workflow-profile extra/workflow/profiles/<site> all
"""

import csv
import sys
from pathlib import Path
from typing import Annotated, TextIO

import typer

from afabench.compute_estimate.planning import (
    InvocationError,
    PlannedJob,
    plan_jobs,
)

ALLOCATION_COLUMNS = ["rule", "device", "cpus", "gpus"]

app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def estimate_compute(
    ctx: typer.Context,
    output: Annotated[
        Path | None,
        typer.Option(help="Write the planned jobs here instead of stdout."),
    ] = None,
) -> None:
    """Plan the jobs of `snakemake <arguments>`; other options go to it."""
    try:
        jobs = plan_jobs(ctx.args)
    except InvocationError:
        raise typer.Exit(1) from None
    if output is None:
        _write_planned_jobs(jobs, sys.stdout)
    else:
        with output.open("w", newline="") as stream:
            _write_planned_jobs(jobs, stream)


def _write_planned_jobs(jobs: list[PlannedJob], stream: TextIO) -> None:
    wildcard_columns = list(
        dict.fromkeys(name for job in jobs for name in job.wildcards)
    )
    writer = csv.DictWriter(
        stream, fieldnames=ALLOCATION_COLUMNS + wildcard_columns
    )
    writer.writeheader()
    for job in jobs:
        writer.writerow(
            {
                "rule": job.rule,
                "device": job.device,
                "cpus": job.cpus,
                "gpus": job.gpus,
                **job.wildcards,
            }
        )


if __name__ == "__main__":
    app()
