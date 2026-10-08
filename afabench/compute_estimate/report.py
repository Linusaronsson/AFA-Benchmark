"""
Report a compute estimate: per-job rows, and totals per group of jobs.

`estimate-compute` prints `format_report` and writes `per_job_table` as
CSV. Totals hold only estimated jobs, as mean and p90 job-hours,
core-hours and GPU-hours; a p90 total adds up each job's p90 job duration.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict

import pandas as pd

from afabench.compute_estimate.estimate import (
    IDENTITY_COLUMNS,
    ComputeEstimate,
    MatchLevel,
)
from afabench.core.job_duration_table import RECORD_FIELD_DTYPES

JOB_COLUMNS = [
    "rule",
    "wildcards",
    *IDENTITY_COLUMNS,
    "device",
    "cpus",
    "gpus",
]
ESTIMATE_COLUMNS = [
    "match_level",
    "matched_job_records",
    "mean_job_duration_seconds",
    "p90_job_duration_seconds",
    "mean_job_hours",
    "p90_job_hours",
    "mean_core_hours",
    "p90_core_hours",
    "mean_gpu_hours",
    "p90_gpu_hours",
]
# As in the job duration table, nullable
PER_JOB_DTYPES = {
    "rule": "string",
    "wildcards": "string",
    **{column: RECORD_FIELD_DTYPES[column] for column in IDENTITY_COLUMNS},
    "device": "string",
    "cpus": "Int64",
    "gpus": "Int64",
    "match_level": "string",
    "matched_job_records": "Int64",
    **dict.fromkeys(ESTIMATE_COLUMNS[2:], "Float64"),
}
GROUPING_COLUMNS = [column for column in JOB_COLUMNS if column != "wildcards"]
DEFAULT_GROUPING = ["stage", "device"]
TOTAL_COLUMNS = ESTIMATE_COLUMNS[4:]
COUNTED_MATCH_LEVELS: list[MatchLevel] = ["exact", "pooled", "unestimated"]
UNESTIMATED_GROUPING = ["stage", "name", "dataset_key", "device"]


class UnknownGroupingColumnError(ValueError):
    """Totals were asked to be grouped by a column jobs do not have."""


def per_job_table(estimate: ComputeEstimate) -> pd.DataFrame:
    """Return one row per planned job: identity, allocation, estimate."""
    rows = [
        {
            "rule": job.job.rule,
            "wildcards": json.dumps(job.job.wildcards),
            **(
                asdict(job.job.identity)
                if job.job.identity is not None
                else dict.fromkeys(IDENTITY_COLUMNS)
            ),
            "device": job.job.device,
            "cpus": job.job.cpus,
            "gpus": job.job.gpus,
            **{column: getattr(job, column) for column in ESTIMATE_COLUMNS},
        }
        for job in estimate.jobs
    ]
    return pd.DataFrame(
        rows, columns=pd.Index([*JOB_COLUMNS, *ESTIMATE_COLUMNS])
    ).astype(PER_JOB_DTYPES)


def check_grouping(by: Sequence[str]) -> None:
    """Raise `UnknownGroupingColumnError` unless jobs have columns `by`."""
    unknown = [column for column in by if column not in GROUPING_COLUMNS]
    if unknown or not by:
        message = (
            f"Cannot group by {unknown or 'no column'}; jobs have the "
            f"columns {GROUPING_COLUMNS}"
        )
        raise UnknownGroupingColumnError(message)


def group_totals(
    estimate: ComputeEstimate, by: Sequence[str] = DEFAULT_GROUPING
) -> pd.DataFrame:
    """
    Total the estimated jobs' hours, mean and p90, per group of jobs.

    Each group also counts its jobs per match level. Unestimated jobs are
    counted but not in the totals, and jobs of rules without job records
    are left out. p90 totals add up each job's p90.
    """
    check_grouping(by)
    jobs = per_job_table(estimate)
    counted = jobs.loc[jobs["match_level"].isin(COUNTED_MATCH_LEVELS)]
    grouped = (
        counted.loc[:, list(by)]
        .assign(
            **{
                level: counted["match_level"].eq(level)
                for level in COUNTED_MATCH_LEVELS
            },
            **{
                column: counted[column].astype(float)
                for column in TOTAL_COLUMNS
            },
        )
        .groupby(list(by), dropna=False, sort=True)
    )
    totals = pd.concat(
        [
            grouped[COUNTED_MATCH_LEVELS].sum(),
            # Groups of only unestimated jobs have no total.
            grouped[TOTAL_COLUMNS].sum(min_count=1),
        ],
        axis=1,
    )
    totals.insert(0, "jobs", totals[COUNTED_MATCH_LEVELS].sum(axis=1))
    return totals.reset_index()


def format_report(
    estimate: ComputeEstimate,
    *,
    source: str,
    by: Sequence[str] = DEFAULT_GROUPING,
) -> str:
    """Return the report `estimate-compute` prints, naming `source`."""
    jobs = per_job_table(estimate)
    counts = jobs["match_level"].value_counts()
    hardware = estimate.hardware
    lines = [
        f"Compute estimate of {_plural(len(jobs), 'planned job')}: "
        + ", ".join(
            f"{counts.get(level, 0)} {level}" for level in COUNTED_MATCH_LEVELS
        )
        + f", {counts.get('no_job_record', 0)} without job records",
        f"Job durations from {source}: "
        f"{_plural(estimate.matched_job_records, 'job record')} matched; "
        f"hosts: {_listing(hardware.hosts)}; "
        f"CPU models: {_listing(hardware.cpu_models)}; "
        f"GPU models: {_listing(hardware.gpu_models)}",
    ]
    refused = estimate.refused_smoke_test_job_records
    if refused:
        lines.append(f"Refused {_plural(refused, 'smoke-test job record')}")
    totals = group_totals(estimate, by)
    overall = {
        **dict.fromkeys(by, ""),
        **totals[["jobs", *COUNTED_MATCH_LEVELS]].sum(),
        **totals[TOTAL_COLUMNS].sum(min_count=1),
    }
    overall[by[0]] = "total"
    totals = pd.concat([totals, pd.DataFrame([overall])], ignore_index=True)
    lines += [
        "",
        f"Totals by {' and '.join(by)} in hours, unestimated jobs not "
        "included:",
        _table(totals),
        "",
    ]
    unestimated = jobs.loc[jobs["match_level"] == "unestimated"]
    if unestimated.empty:
        lines.append("Unestimated jobs: none")
    else:
        lines += [
            "Unestimated jobs, not in the totals:",
            _table(
                unestimated.loc[:, UNESTIMATED_GROUPING]
                .value_counts(dropna=False)
                .sort_index()
                .reset_index(name="jobs")
            ),
        ]
    lines.append("")
    without_records = jobs.loc[
        jobs["match_level"] == "no_job_record", "rule"
    ].value_counts(sort=False)
    if not without_records.empty:
        lines.append(
            "Not estimated, rules without job records: "
            + ", ".join(
                f"{rule} ({count})" for rule, count in without_records.items()
            )
        )
    unknown_cpus = (
        jobs["mean_core_hours"].isna() & jobs["mean_job_hours"].notna()
    )
    if unknown_cpus.any():
        lines.append(
            f"Core-hours leave out {_plural(int(unknown_cpus.sum()), 'job')}"
            " whose CPUs the cluster chooses."
        )
    return "\n".join(lines) + "\n"


def _table(table: pd.DataFrame) -> str:
    formatted = pd.DataFrame(
        {
            column: values.map("{:.2f}".format, na_action="ignore")
            if pd.api.types.is_float_dtype(values)
            else values
            for column, values in table.items()
        }
    )
    shown = formatted.astype(object)
    return shown.where(shown.notna(), "-").to_string(index=False)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _listing(values: list[str]) -> str:
    return ", ".join(values) or "unknown"
