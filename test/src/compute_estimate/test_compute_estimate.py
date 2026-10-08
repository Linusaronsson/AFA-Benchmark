"""
The compute estimate of planned jobs from measured job durations (#83).

Job durations come from job records written under an output root and
loaded as the job duration table, as `estimate-compute` loads them.
"""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from afabench.compute_estimate.estimate import estimate_compute
from afabench.compute_estimate.planning import PlannedJob
from afabench.core.job_duration_table import load_job_duration_table
from afabench.core.job_record import (
    JOB_RECORD_VERSION,
    ExitStatus,
    JobIdentity,
    JobRecord,
)

ALPHA_TRAINING = JobIdentity(
    stage="training",
    name="alpha",
    dataset_key="cube",
    dataset_realization_index=0,
    train_seed=0,
    train_hard_budget=1,
)


class JobRecords:
    """An output root to write job records with chosen job durations to."""

    output_root: Path
    count: int

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        output_root.mkdir()
        self.count = 0

    def add(
        self,
        identity: JobIdentity,
        job_duration_seconds: float,
        *,
        device: str = "cuda",
        cpus: int | None = 4,
        gpus: int = 1,
        exit_status: ExitStatus = "completed",
        smoke_test: bool = False,
    ) -> None:
        self.count += 1
        record = JobRecord(
            job_record_version=JOB_RECORD_VERSION,
            stage=identity.stage,
            name=identity.name,
            dataset_key=identity.dataset_key,
            dataset_realization_index=identity.dataset_realization_index,
            pretrain_seed=identity.pretrain_seed,
            train_seed=identity.train_seed,
            eval_seed=identity.eval_seed,
            train_hard_budget=identity.train_hard_budget,
            train_soft_budget_param=identity.train_soft_budget_param,
            eval_hard_budget=identity.eval_hard_budget,
            eval_soft_budget_param=identity.eval_soft_budget_param,
            eval_batch_size=identity.eval_batch_size,
            started_at="2026-10-08T12:00:00+00:00",
            ended_at="2026-10-08T13:00:00+00:00",
            job_duration_seconds=job_duration_seconds,
            exit_status=exit_status,
            exit_code=0 if exit_status == "completed" else 1,
            device="cuda" if device == "cuda" else "cpu",
            cpus=cpus,
            gpus=gpus,
            time_limit_minutes=600,
            gpu_model="NVIDIA A40" if gpus else None,
            cpu_model="AMD EPYC 7742",
            host=f"node{self.count % 2}",
            slurm_job_id=str(self.count),
            code_commit="0123abc",
            smoke_test=smoke_test,
        )
        path = self.output_root / f"job{self.count}.job_record.json"
        path.write_text(json.dumps(record.to_json_dict()))

    def table(self) -> pd.DataFrame:
        return load_job_duration_table(self.output_root)


@pytest.fixture
def records(tmp_path: Path) -> JobRecords:
    return JobRecords(tmp_path / "output")


def planned(
    identity: JobIdentity | None,
    *,
    rule: str = "train_method",
    device: str = "cuda",
    cpus: int | None = 8,
    gpus: int = 1,
) -> PlannedJob:
    return PlannedJob(
        rule=rule,
        wildcards={},
        identity=identity,
        device="cuda" if device == "cuda" else "cpu",
        cpus=cpus,
        gpus=gpus,
    )


def test_a_job_with_a_record_of_the_same_identity_and_device_matches_exactly(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 7200)
    records.add(replace(ALPHA_TRAINING, train_seed=1), 3600)

    (job,) = estimate_compute([planned(ALPHA_TRAINING)], records.table()).jobs

    assert job.match_level == "exact"
    assert job.matched_job_records == 1
    assert job.mean_job_duration_seconds == 7200
    assert job.p90_job_duration_seconds == 7200


def test_a_job_without_an_exact_match_pools_its_stage_name_dataset_and_device(
    records: JobRecords,
) -> None:
    # Other seeds, dataset realizations, hard budgets and soft-budget
    # parameters of the same method on the same dataset and device
    for seconds, seed in zip(range(10, 101, 10), range(1, 11), strict=True):
        records.add(
            replace(
                ALPHA_TRAINING,
                train_seed=seed,
                dataset_realization_index=seed % 3,
                train_hard_budget=None,
                train_soft_budget_param=seed / 10,
            ),
            seconds,
        )

    (job,) = estimate_compute([planned(ALPHA_TRAINING)], records.table()).jobs

    assert job.match_level == "pooled"
    assert job.matched_job_records == 10
    assert job.mean_job_duration_seconds == 55
    # Linear interpolation between the 9th and 10th of 10 durations
    assert job.p90_job_duration_seconds == pytest.approx(91)


@pytest.mark.parametrize(
    ("identity", "device", "gpus"),
    [
        (ALPHA_TRAINING, "cpu", 0),
        (replace(ALPHA_TRAINING, dataset_key="mnist"), "cuda", 1),
    ],
    ids=["other device", "other dataset key"],
)
def test_matching_never_crosses_device_or_dataset_key(
    records: JobRecords, identity: JobIdentity, device: str, gpus: int
) -> None:
    records.add(identity, 3600, device=device, gpus=gpus)

    (job,) = estimate_compute([planned(ALPHA_TRAINING)], records.table()).jobs

    assert job.match_level == "unestimated"
    assert job.matched_job_records == 0
    assert job.mean_job_duration_seconds is None
    assert job.p90_job_duration_seconds is None


@pytest.mark.parametrize("exit_status", ["failed", "timeout"])
def test_failed_and_timed_out_jobs_give_no_job_durations(
    records: JobRecords, exit_status: ExitStatus
) -> None:
    records.add(ALPHA_TRAINING, 60, exit_status=exit_status)
    records.add(replace(ALPHA_TRAINING, train_seed=1), 3600)

    (job,) = estimate_compute([planned(ALPHA_TRAINING)], records.table()).jobs

    assert job.match_level == "pooled"
    assert job.mean_job_duration_seconds == 3600


def test_smoke_test_job_durations_are_refused(records: JobRecords) -> None:
    records.add(ALPHA_TRAINING, 1, smoke_test=True)
    records.add(replace(ALPHA_TRAINING, train_seed=1), 2, smoke_test=True)

    estimate = estimate_compute([planned(ALPHA_TRAINING)], records.table())

    assert [job.match_level for job in estimate.jobs] == ["unestimated"]
    assert estimate.refused_smoke_test_job_records == 2


def test_core_and_gpu_hours_follow_the_planned_allocation_not_the_measured(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 3600, cpus=4, gpus=1)
    records.add(ALPHA_TRAINING, 10800, cpus=4, gpus=1)

    (job,) = estimate_compute(
        [planned(ALPHA_TRAINING, cpus=8, gpus=2)], records.table()
    ).jobs

    assert (job.mean_job_hours, job.p90_job_hours) == (2, pytest.approx(2.8))
    assert (job.mean_core_hours, job.p90_core_hours) == (
        16,
        pytest.approx(22.4),
    )
    assert (job.mean_gpu_hours, job.p90_gpu_hours) == (4, pytest.approx(5.6))


def test_core_hours_are_unknown_when_the_cluster_chooses_the_cpus(
    records: JobRecords,
) -> None:
    records.add(ALPHA_TRAINING, 3600)

    (job,) = estimate_compute(
        [planned(ALPHA_TRAINING, cpus=None, gpus=1)], records.table()
    ).jobs

    assert job.mean_core_hours is None
    assert job.p90_core_hours is None
    assert job.mean_gpu_hours == 1


def test_jobs_of_rules_without_job_records_are_not_unestimated(
    records: JobRecords,
) -> None:
    aggregation = planned(None, rule="merge_eval_perf", device="cpu", gpus=0)

    (job,) = estimate_compute([aggregation], records.table()).jobs

    assert job.match_level == "no_job_record"
    assert job.mean_job_duration_seconds is None
