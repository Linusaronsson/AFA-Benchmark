"""
Estimate the compute of planned jobs from measured job durations.

The second half of a compute estimate
(`docs/adr/0006-job-records-beside-artifacts.md`): each planned job is
matched to the completed job records of a job duration table, in a fixed
fallback order. An exact match has the same identity and device. Otherwise
the job is pooled with every job of its stage, name, dataset key and
device, whatever their seeds, dataset realizations, hard budgets and
soft-budget parameters. Otherwise it is unestimated. Matching never crosses
device or dataset key, and durations are not normalized across hardware.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Literal

import pandas as pd

from afabench.compute_estimate.planning import PlannedJob
from afabench.core.job_record import JobIdentity

# no_job_record: the job's rule writes no job record, so no job duration
# can match it; aggregation and visualization jobs are not computational.
type MatchLevel = Literal["exact", "pooled", "unestimated", "no_job_record"]

IDENTITY_COLUMNS = [field.name for field in fields(JobIdentity)]
EXACT_COLUMNS = [*IDENTITY_COLUMNS, "device"]
POOL_COLUMNS = ["stage", "name", "dataset_key", "device"]

# The values of a job's identity columns and device, None for null
type Key = tuple[object, ...]


@dataclass(frozen=True, kw_only=True)
class JobEstimate:
    """A planned job and the job durations it was matched to."""

    job: PlannedJob
    match_level: MatchLevel
    matched_job_records: int
    # None when unestimated
    mean_job_duration_seconds: float | None
    p90_job_duration_seconds: float | None

    @property
    def mean_job_hours(self) -> float | None:
        return _hours(self.mean_job_duration_seconds, 1)

    @property
    def p90_job_hours(self) -> float | None:
        return _hours(self.p90_job_duration_seconds, 1)

    # Core-hours and GPU-hours use the planned allocation: the measured
    # job may have run with another one.
    @property
    def mean_core_hours(self) -> float | None:
        return _hours(self.mean_job_duration_seconds, self.job.cpus)

    @property
    def p90_core_hours(self) -> float | None:
        return _hours(self.p90_job_duration_seconds, self.job.cpus)

    @property
    def mean_gpu_hours(self) -> float | None:
        return _hours(self.mean_job_duration_seconds, self.job.gpus)

    @property
    def p90_gpu_hours(self) -> float | None:
        return _hours(self.p90_job_duration_seconds, self.job.gpus)


@dataclass(frozen=True, kw_only=True)
class ComputeEstimate:
    jobs: list[JobEstimate]
    # Job records of smoke tests, never used as job durations
    refused_smoke_test_job_records: int


def estimate_compute(
    planned_jobs: Sequence[PlannedJob], job_durations: pd.DataFrame
) -> ComputeEstimate:
    """
    Estimate planned jobs from a job duration table's rows.

    `job_durations` has the columns of `load_job_duration_table`.
    """
    # Only completed jobs ran for their whole job duration. Smoke-test jobs
    # did too, but would make the estimate far too low.
    smoke_test = job_durations["smoke_test"].fillna(value=False).astype(bool)
    usable = job_durations.loc[
        job_durations["exit_status"].eq("completed") & ~smoke_test
    ]
    exact: defaultdict[Key, list[float]] = defaultdict(list)
    pools: defaultdict[Key, list[float]] = defaultdict(list)
    for record, duration in zip(
        _records(usable[EXACT_COLUMNS]),
        usable["job_duration_seconds"].astype(float),
        strict=True,
    ):
        exact[_key(record, EXACT_COLUMNS)].append(duration)
        pools[_key(record, POOL_COLUMNS)].append(duration)
    return ComputeEstimate(
        jobs=[_estimate(job, exact, pools) for job in planned_jobs],
        refused_smoke_test_job_records=int(smoke_test.sum()),
    )


def _estimate(
    job: PlannedJob,
    exact: Mapping[Key, list[float]],
    pools: Mapping[Key, list[float]],
) -> JobEstimate:
    if job.identity is None:
        return _not_estimated(job, "no_job_record")
    values = {**asdict(job.identity), "device": job.device}
    match_level: MatchLevel
    if durations := exact.get(_key(values, EXACT_COLUMNS)):
        match_level = "exact"
    elif durations := pools.get(_key(values, POOL_COLUMNS)):
        match_level = "pooled"
    else:
        return _not_estimated(job, "unestimated")
    series = pd.Series(durations, dtype="float64")
    return JobEstimate(
        job=job,
        match_level=match_level,
        matched_job_records=len(series),
        mean_job_duration_seconds=float(series.mean()),
        p90_job_duration_seconds=float(series.quantile(0.9)),
    )


def _not_estimated(job: PlannedJob, match_level: MatchLevel) -> JobEstimate:
    return JobEstimate(
        job=job,
        match_level=match_level,
        matched_job_records=0,
        mean_job_duration_seconds=None,
        p90_job_duration_seconds=None,
    )


def _hours(seconds: float | None, count: int | None) -> float | None:
    if seconds is None or count is None:
        return None
    return seconds / 3600 * count


def _records(table: pd.DataFrame) -> list[dict[str, object]]:
    """Return a table's rows with None for NA, as job records hold it."""
    values = table.astype(object)
    return values.where(values.notna(), None).to_dict("records")


def _key(values: Mapping[str, object], columns: list[str]) -> Key:
    return tuple(values[column] for column in columns)
