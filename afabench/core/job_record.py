"""
The job record: one pipeline job's identity, job duration and allocation.

Design: `docs/adr/0006-job-records-beside-artifacts.md`; schema:
`docs/reference/job_records.md`. A pipeline rule runs its script through
this module's command, which times the script and writes the record as JSON
beside the artifact the job produced. `null` always means "unknown" or "not
part of this job's identity", never a default. The command runs once per
pipeline job before its script, so this module imports nothing heavy.

    python -m afabench.core.job_record --record <path> --stage <stage>
        [identity and allocation options] -- <script command>
"""

import json
import os
import platform
import re
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from afabench.core.code_identity import AFABENCH_CHECKOUT, code_identity

JOB_RECORD_VERSION = 1
# A record is named after its artifact: model.bundle, model.job_record.json
JOB_RECORD_SUFFIX = ".job_record.json"

# Plain assignments, not `type` statements: typer reads only these as
# choices. The pipeline stages whose jobs are computational and leave a job
# record:
Stage = Literal[
    "dataset_generation",
    "classifier_training",
    "pretraining",
    "training",
    "evaluation",
    "transformation",
]
Device = Literal["cpu", "cuda"]
type ExitStatus = Literal["completed", "failed"]


@dataclass(frozen=True, kw_only=True)
class JobIdentity:
    """What a job computed. Fields outside the stage's identity are null."""

    stage: Stage
    name: str | None = None
    dataset_key: str | None = None
    dataset_realization_index: int | None = None
    pretrain_seed: int | None = None
    train_seed: int | None = None
    eval_seed: int | None = None
    train_hard_budget: int | None = None
    train_soft_budget_param: float | None = None
    eval_hard_budget: int | None = None
    eval_soft_budget_param: float | None = None
    eval_batch_size: int | None = None


@dataclass(frozen=True, kw_only=True)
class Allocation:
    """The hardware a job was allocated, as the pipeline resolved it."""

    device: Device
    cpus: int | None
    gpus: int | None


@dataclass(frozen=True, kw_only=True)
class JobRecord:
    job_record_version: int
    stage: Stage
    name: str | None
    dataset_key: str | None
    dataset_realization_index: int | None
    pretrain_seed: int | None
    train_seed: int | None
    eval_seed: int | None
    train_hard_budget: int | None
    train_soft_budget_param: float | None
    eval_hard_budget: int | None
    eval_soft_budget_param: float | None
    eval_batch_size: int | None
    started_at: str
    ended_at: str
    job_duration_seconds: float
    exit_status: ExitStatus
    exit_code: int | None
    device: Device
    cpus: int | None
    gpus: int | None
    gpu_model: str | None
    cpu_model: str | None
    host: str | None
    slurm_job_id: str | None
    code_commit: str | None
    smoke_test: bool

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_job(
    command: list[str],
    *,
    identity: JobIdentity,
    allocation: Allocation,
    smoke_test: bool,
    record_path: Path,
) -> JobRecord:
    """Run a job's script command, then write and return its job record."""
    started_at = datetime.now(UTC)
    start = time.perf_counter()
    exit_code = subprocess.run(command, check=False).returncode
    job_duration_seconds = time.perf_counter() - start
    ended_at = datetime.now(UTC)
    record = JobRecord(
        job_record_version=JOB_RECORD_VERSION,
        **asdict(identity),
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        job_duration_seconds=job_duration_seconds,
        exit_status="completed" if exit_code == 0 else "failed",
        exit_code=exit_code,
        **asdict(allocation),
        gpu_model=_gpu_model() if allocation.gpus else None,
        cpu_model=_cpu_model(),
        host=socket.gethostname() or None,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        code_commit=code_identity(AFABENCH_CHECKOUT)[0],
        smoke_test=smoke_test,
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record.to_json_dict(), indent=2) + "\n")
    return record


def _gpu_model() -> str | None:
    """Return the names of the GPUs the job sees, null when not queryable."""
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
        return None
    names = dict.fromkeys(
        line.strip() for line in completed.stdout.splitlines() if line.strip()
    )
    return ", ".join(names) or None


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        match = re.search(
            r"^model name\s*:\s*(.+)$", cpuinfo.read_text(), re.MULTILINE
        )
        if match:
            return match.group(1).strip()
    return platform.processor() or None


app = typer.Typer(add_completion=False)


@app.command()
def main(
    command: Annotated[
        list[str], typer.Argument(help="The job's script command, after --.")
    ],
    record: Annotated[
        Path, typer.Option(help="Where to write the job record.")
    ],
    stage: Annotated[Stage, typer.Option()],
    device: Annotated[Device, typer.Option()],
    cpus: Annotated[int | None, typer.Option()] = None,
    gpus: Annotated[int | None, typer.Option()] = None,
    smoke_test: Annotated[bool, typer.Option()] = False,  # noqa: FBT002
    name: Annotated[str | None, typer.Option()] = None,
    dataset_key: Annotated[str | None, typer.Option()] = None,
    dataset_realization_index: Annotated[int | None, typer.Option()] = None,
    pretrain_seed: Annotated[int | None, typer.Option()] = None,
    train_seed: Annotated[int | None, typer.Option()] = None,
    eval_seed: Annotated[int | None, typer.Option()] = None,
    train_hard_budget: Annotated[int | None, typer.Option()] = None,
    train_soft_budget_param: Annotated[float | None, typer.Option()] = None,
    eval_hard_budget: Annotated[int | None, typer.Option()] = None,
    eval_soft_budget_param: Annotated[float | None, typer.Option()] = None,
    eval_batch_size: Annotated[int | None, typer.Option()] = None,
    time_file: Annotated[
        Path | None,
        typer.Option(help="Also write the job duration in seconds here."),
    ] = None,
) -> None:
    """Run a pipeline job's script and write its job record."""
    job_record = run_job(
        command,
        identity=JobIdentity(
            stage=stage,
            name=name,
            dataset_key=dataset_key,
            dataset_realization_index=dataset_realization_index,
            pretrain_seed=pretrain_seed,
            train_seed=train_seed,
            eval_seed=eval_seed,
            train_hard_budget=train_hard_budget,
            train_soft_budget_param=train_soft_budget_param,
            eval_hard_budget=eval_hard_budget,
            eval_soft_budget_param=eval_soft_budget_param,
            eval_batch_size=eval_batch_size,
        ),
        allocation=Allocation(device=device, cpus=cpus, gpus=gpus),
        smoke_test=smoke_test,
        record_path=record,
    )
    if job_record.exit_code != 0:
        raise typer.Exit(job_record.exit_code or 1)
    # The time aggregation still reads *_time.txt (ADR-0006 expand step)
    if time_file is not None:
        time_file.write_text(f"{job_record.job_duration_seconds:.6f}\n")


if __name__ == "__main__":
    app()
