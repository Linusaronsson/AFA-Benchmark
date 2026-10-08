"""
Estimate the compute of a pipeline invocation (`just estimate-compute`).

Takes the same Snakemake arguments as the real run (`--profile`,
`--workflow-profile`, `--config`, `--set-resources`, target, ...), plans
the jobs that invocation would run, matches each to job durations and
prints the compute estimate: job-hours, core-hours and GPU-hours, mean and
p90, per pipeline stage and device. See `afabench.compute_estimate` and
`docs/how-to/estimate_compute.md`.

Options, before or among the Snakemake arguments:
    --job-durations PATH  A job duration table, or an output root of job
                          records (default: extra/output).
    --by COLUMN           Group the totals by these job columns instead,
                          repeatable (e.g. --by name --by dataset_key).
    --output CSV          Also write each planned job's estimate here.
    --strict              Exit with 1 when a planned job is unestimated.

Usage:
    just estimate-compute \
        --profile extra/workflow/profiles/config/kdd26 \
        --workflow-profile extra/workflow/profiles/<site> all
    just estimate-compute \
        --job-durations extra/release_job_duration_table.parquet \
        --by name --output estimate.csv <Snakemake arguments>
"""

from pathlib import Path
from typing import Annotated

import typer

from afabench.compute_estimate.estimate import estimate_compute
from afabench.compute_estimate.planning import InvocationError, plan_jobs
from afabench.compute_estimate.report import (
    DEFAULT_GROUPING,
    UnknownGroupingColumnError,
    check_grouping,
    format_report,
    per_job_table,
)
from afabench.core.job_duration_table import (
    empty_job_duration_table,
    load_job_duration_table,
)
from afabench.release.manifest import (
    JOB_DURATION_TABLE_FILENAME,
    RELEASE_MANIFEST_FILENAME,
    read_release_manifest,
)

DEFAULT_JOB_DURATIONS = Path("extra/output")

app = typer.Typer()


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def main(
    ctx: typer.Context,
    job_durations: Annotated[
        Path,
        typer.Option(
            help=(
                "A job duration table, or an output root of job records, "
                "such as a downloaded release's."
            )
        ),
    ] = DEFAULT_JOB_DURATIONS,
    by: Annotated[
        list[str] | None,
        typer.Option(
            help=(
                "Group the totals by this job column; repeat for several. "
                f"Default: {' and '.join(DEFAULT_GROUPING)}."
            )
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Write each planned job's estimate here as CSV."),
    ] = None,
    strict: Annotated[  # noqa: FBT002
        bool,
        typer.Option(help="Exit with 1 when a planned job is unestimated."),
    ] = False,
) -> None:
    """Estimate `snakemake <arguments>`; other options go to Snakemake."""
    grouping = by or DEFAULT_GROUPING
    try:
        check_grouping(grouping)
    except UnknownGroupingColumnError as error:
        raise typer.BadParameter(str(error), param_hint="--by") from None
    if job_durations.exists():
        table = load_job_duration_table(job_durations)
    elif job_durations == DEFAULT_JOB_DURATIONS:
        # Nothing has run here yet: every job is unestimated.
        table = empty_job_duration_table()
    else:
        message = f"No job duration table or output root at {job_durations}"
        raise typer.BadParameter(message, param_hint="--job-durations")
    try:
        jobs = plan_jobs(ctx.args)
    except InvocationError:
        raise typer.Exit(1) from None
    estimate = estimate_compute(jobs, table)
    if output is not None:
        per_job_table(estimate).to_csv(output, index=False)
    typer.echo(
        format_report(
            estimate, source=_described_source(job_durations), by=grouping
        ),
        nl=False,
    )
    unestimated = sum(
        job.match_level == "unestimated" for job in estimate.jobs
    )
    if strict and unestimated:
        noun = "job is" if unestimated == 1 else "jobs are"
        typer.echo(
            f"--strict: {unestimated} planned {noun} unestimated", err=True
        )
        raise typer.Exit(1)


def _described_source(job_durations: Path) -> str:
    """Name the release of a restored release's job duration table."""
    manifest_path = job_durations.parent / RELEASE_MANIFEST_FILENAME
    if (
        job_durations.name != JOB_DURATION_TABLE_FILENAME
        or not manifest_path.is_file()
    ):
        return str(job_durations)
    manifest = read_release_manifest(manifest_path)
    entry = manifest.job_duration_table
    # Downloading another release with --overwrite can leave an earlier
    # release's table beside the new manifest.
    if entry is None or entry.size_bytes != job_durations.stat().st_size:
        return (
            f"{job_durations} (not the table of release "
            f"{manifest.release_id}, whose manifest is beside it)"
        )
    return (
        f"{job_durations} (release {manifest.release_id}, "
        f"{manifest.scope} scope)"
    )


if __name__ == "__main__":
    app()
