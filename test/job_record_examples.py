"""Job records for tests that need them on disk without running a job."""

import json
from dataclasses import asdict, replace
from pathlib import Path

from afabench.core.job_record import JOB_RECORD_VERSION, JobIdentity, JobRecord


def job_record(identity: JobIdentity, **fields: object) -> JobRecord:
    """Return a completed CPU job's record of `identity`, `fields` replaced."""
    record = JobRecord(
        job_record_version=JOB_RECORD_VERSION,
        **asdict(identity),
        started_at="2026-10-08T12:00:00+00:00",
        ended_at="2026-10-08T12:01:00+00:00",
        job_duration_seconds=60.0,
        exit_status="completed",
        exit_code=0,
        device="cpu",
        cpus=1,
        gpus=0,
        time_limit_minutes=None,
        gpu_model=None,
        cpu_model=None,
        host=None,
        slurm_job_id=None,
        code_commit=None,
        smoke_test=False,
    )
    return replace(record, **fields)


def write_job_record(path: Path, record: JobRecord) -> None:
    """Write `record` at `path` as the job record wrapper would."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_json_dict(), indent=2) + "\n")
