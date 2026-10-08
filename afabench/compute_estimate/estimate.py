"""
Estimate the compute of planned jobs from measured job durations.

The second half of a compute estimate
(`docs/adr/0007-job-records-beside-artifacts.md`): each planned job is
matched to the completed job records of a job duration table, in a fixed
fallback order. An exact match has the same identity and device. Otherwise
the job is pooled with every job of its stage, name, dataset key and
device, whatever their seeds, dataset realizations, hard budgets and
soft-budget parameters. Otherwise it is unestimated. Matching never crosses
device or dataset key, and durations are not normalized across hardware.

Failed and timed-out job records give no job duration. Instead, each
planned job type (stage, name and dataset key) that has them gets a failure
history: its jobs may fail or hit their time limit again, so their estimate
may be low.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, astuple, dataclass, fields
from typing import Literal

import pandas as pd

from afabench.compute_estimate.planning import PlannedJob
from afabench.core.job_record import JobIdentity, JobType

# no_job_record: the job's rule writes no job record, so no job duration
# can match it; aggregation and visualization jobs are not computational.
type MatchLevel = Literal["exact", "pooled", "unestimated", "no_job_record"]

IDENTITY_COLUMNS = [field.name for field in fields(JobIdentity)]
EXACT_COLUMNS = [*IDENTITY_COLUMNS, "device"]
JOB_TYPE_COLUMNS = [field.name for field in fields(JobType)]
POOL_COLUMNS = [*JOB_TYPE_COLUMNS, "device"]

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
class Hardware:
    """Where matched job durations were measured, distinct and sorted."""

    hosts: list[str]
    cpu_models: list[str]
    gpu_models: list[str]


@dataclass(frozen=True, kw_only=True)
class FailureHistory:
    """The failed and timed-out job records of a planned job type."""

    job_type: JobType
    failed: int
    timed_out: int
    # Distinct known time limits of the timed-out jobs, sorted
    time_limits_minutes: list[int]


@dataclass(frozen=True, kw_only=True)
class ComputeEstimate:
    jobs: list[JobEstimate]
    # Job records some job was matched to
    matched_job_records: int
    hardware: Hardware
    # Job records of smoke tests, never used as job durations
    refused_smoke_test_job_records: int
    # Sorted by job type
    failure_histories: list[FailureHistory]


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
    ].reset_index(drop=True)
    # Positions of the usable job records per key
    exact: defaultdict[Key, list[int]] = defaultdict(list)
    pools: defaultdict[Key, list[int]] = defaultdict(list)
    for position, record in enumerate(_records(usable[EXACT_COLUMNS])):
        exact[_key(record, EXACT_COLUMNS)].append(position)
        pools[_key(record, POOL_COLUMNS)].append(position)
    matches = [_match(job, exact, pools) for job in planned_jobs]
    durations = usable["job_duration_seconds"].astype(float)
    matched = usable.iloc[
        sorted(
            {position for _, positions in matches for position in positions}
        )
    ]
    return ComputeEstimate(
        jobs=[
            _estimate(job, match_level, durations.iloc[positions])
            for job, (match_level, positions) in zip(
                planned_jobs, matches, strict=True
            )
        ],
        matched_job_records=len(matched),
        hardware=Hardware(
            hosts=_distinct(matched["host"]),
            cpu_models=_distinct(matched["cpu_model"]),
            gpu_models=_distinct(matched["gpu_model"]),
        ),
        refused_smoke_test_job_records=int(smoke_test.sum()),
        failure_histories=_failure_histories(
            planned_jobs,
            job_durations.loc[
                job_durations["exit_status"].isin(["failed", "timeout"])
                & ~smoke_test
            ],
        ),
    )


def _failure_histories(
    planned_jobs: Sequence[PlannedJob], failures: pd.DataFrame
) -> list[FailureHistory]:
    """Return the failure history of each planned job type that has one."""
    planned_types = {
        job.job_type for job in planned_jobs if job.job_type is not None
    }
    timed_out: defaultdict[Key, int] = defaultdict(int)
    failed: defaultdict[Key, int] = defaultdict(int)
    time_limits: defaultdict[Key, set[int]] = defaultdict(set)
    for record in _records(
        failures.loc[
            :, [*JOB_TYPE_COLUMNS, "exit_status", "time_limit_minutes"]
        ]
    ):
        key = _key(record, JOB_TYPE_COLUMNS)
        if record["exit_status"] == "timeout":
            timed_out[key] += 1
            if isinstance(limit := record["time_limit_minutes"], int):
                time_limits[key].add(limit)
        else:
            failed[key] += 1
    histories = []
    for job_type in sorted(
        planned_types, key=lambda job_type: tuple(map(str, astuple(job_type)))
    ):
        key = astuple(job_type)
        if failed[key] or timed_out[key]:
            histories.append(
                FailureHistory(
                    job_type=job_type,
                    failed=failed[key],
                    timed_out=timed_out[key],
                    time_limits_minutes=sorted(time_limits[key]),
                )
            )
    return histories


def _match(
    job: PlannedJob,
    exact: Mapping[Key, list[int]],
    pools: Mapping[Key, list[int]],
) -> tuple[MatchLevel, list[int]]:
    """Return the match level and the matched job records' positions."""
    if job.identity is None:
        return "no_job_record", []
    values = {**asdict(job.identity), "device": job.device}
    if positions := exact.get(_key(values, EXACT_COLUMNS)):
        return "exact", positions
    if positions := pools.get(_key(values, POOL_COLUMNS)):
        return "pooled", positions
    return "unestimated", []


def _estimate(
    job: PlannedJob, match_level: MatchLevel, durations: pd.Series
) -> JobEstimate:
    estimated = not durations.empty
    return JobEstimate(
        job=job,
        match_level=match_level,
        matched_job_records=len(durations),
        mean_job_duration_seconds=float(durations.mean())
        if estimated
        else None,
        p90_job_duration_seconds=float(durations.quantile(0.9))
        if estimated
        else None,
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


def _distinct(values: pd.Series) -> list[str]:
    return sorted(str(value) for value in values.dropna().unique())
