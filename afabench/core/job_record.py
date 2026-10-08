"""
The job record: one pipeline job's identity, job duration and allocation.

Design: `docs/adr/0007-job-records-beside-artifacts.md`; schema:
`docs/reference/job_records.md`. A pipeline rule runs its script through
this module's command, which times the script and writes the record as JSON
beside the artifact the job produced. `null` always means "unknown" or "not
part of this job's identity", never a default. The command runs once per
pipeline job before its script, so this module imports nothing heavy.

    python -m afabench.core.job_record --record <path>
        --failed-record <path> --stage <stage>
        [identity and allocation options] -- <script command>

Snakemake deletes a failed job's declared outputs, so the record of a job
that failed or timed out goes to the undeclared failed-record path instead,
with an attempt id that keeps repeated attempts apart.
"""

import json
import os
import platform
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from afabench.core.code_identity import AFABENCH_CHECKOUT, code_identity

JOB_RECORD_VERSION = 1
# A record is named after its artifact: model.bundle, model.job_record.json
JOB_RECORD_SUFFIX = ".job_record.json"
# SLURM sends SIGTERM at a job's time limit and SIGKILL after its KillWait,
# 30 seconds by default. Kill a script that ignores SIGTERM before then, so
# that the record is still written.
SCRIPT_KILL_DELAY_SECONDS = 10

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
type ExitStatus = Literal["completed", "failed", "timeout"]


@dataclass(frozen=True, kw_only=True)
class JobType:
    """What jobs share when they differ only in seeds, realization, budgets."""

    stage: Stage
    name: str | None
    dataset_key: str | None


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

    @property
    def job_type(self) -> JobType:
        return JobType(
            stage=self.stage, name=self.name, dataset_key=self.dataset_key
        )


@dataclass(frozen=True, kw_only=True)
class Allocation:
    """The hardware a job was allocated, as the pipeline resolved it."""

    device: Device
    cpus: int | None
    gpus: int | None
    time_limit_minutes: int | None


def allocated_cpus(
    resources: Mapping[str, object], threads: int
) -> int | None:
    """
    Return the CPUs a job requests, as the SLURM executor resolves them.

    `resources` are the job's final Snakemake resources. None when a
    negative `cpus_per_task` leaves the CPUs to the cluster's default.
    """
    cpus_per_task = resources.get("cpus_per_task")
    if not cpus_per_task:
        return threads
    if not isinstance(cpus_per_task, int):
        message = f"cpus_per_task must be an integer, got {cpus_per_task!r}"
        raise TypeError(message)
    return None if cpus_per_task < 0 else max(1, cpus_per_task)


def allocated_gpus(resources: Mapping[str, object]) -> int:
    """Return the GPUs a job requests through `gpu` or a GPU `gres`."""
    # The SLURM executor submits any set `gpu` as --gpus=<gpu>.
    gpu = resources.get("gpu")
    if gpu:
        return int(str(gpu))
    match = re.fullmatch(r"gpu(?::\w+)?:(\d+)", str(resources.get("gres", "")))
    return int(match[1]) if match else 0


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
    time_limit_minutes: int | None
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
    failed_record_path: Path,
) -> JobRecord:
    """
    Run a job's script command, then write and return its job record.

    A completed job's record goes to `record_path`. A failed or timed-out
    job's goes beside `failed_record_path`, named uniquely per attempt.
    """
    started_at = datetime.now(UTC)
    start = time.perf_counter()
    process = subprocess.Popen(command)
    terminated = threading.Event()
    kill = threading.Timer(SCRIPT_KILL_DELAY_SECONDS, process.kill)
    kill.daemon = True

    def terminate(_signal: int, _frame: object) -> None:
        if terminated.is_set():
            return
        terminated.set()
        # SLURM signals the whole job step, but a local SIGTERM reaches only
        # this process.
        process.terminate()
        kill.start()

    previous_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        exit_code = process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        kill.cancel()
    job_duration_seconds = time.perf_counter() - start
    ended_at = datetime.now(UTC)
    exit_status: ExitStatus = (
        "timeout"
        if terminated.is_set()
        else "completed"
        if exit_code == 0
        else "failed"
    )
    record = JobRecord(
        job_record_version=JOB_RECORD_VERSION,
        **asdict(identity),
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
        job_duration_seconds=job_duration_seconds,
        exit_status=exit_status,
        exit_code=exit_code,
        **asdict(allocation),
        gpu_model=_gpu_model() if allocation.gpus else None,
        cpu_model=_cpu_model(),
        host=socket.gethostname() or None,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        code_commit=code_identity(AFABENCH_CHECKOUT)[0],
        smoke_test=smoke_test,
    )
    if exit_status != "completed":
        record_path = _attempt_path(failed_record_path, started_at)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record.to_json_dict(), indent=2) + "\n")
    return record


def _attempt_path(failed_record_path: Path, started_at: datetime) -> Path:
    """Insert a unique, chronologically sortable attempt id before the suffix."""
    stem = failed_record_path.name.removesuffix(JOB_RECORD_SUFFIX)
    attempt = f"{started_at:%Y%m%dT%H%M%S%fZ}-{uuid.uuid4().hex[:8]}"
    return failed_record_path.with_name(f"{stem}.{attempt}{JOB_RECORD_SUFFIX}")


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
        Path, typer.Option(help="Where to write a completed job's record.")
    ],
    failed_record: Annotated[
        Path,
        typer.Option(
            help=(
                "Where to write a failed or timed-out job's record, with an "
                "attempt id inserted before .job_record.json."
            )
        ),
    ],
    stage: Annotated[Stage, typer.Option()],
    device: Annotated[Device, typer.Option()],
    cpus: Annotated[int | None, typer.Option()] = None,
    gpus: Annotated[int | None, typer.Option()] = None,
    time_limit_minutes: Annotated[int | None, typer.Option()] = None,
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
        allocation=Allocation(
            device=device,
            cpus=cpus,
            gpus=gpus,
            time_limit_minutes=time_limit_minutes,
        ),
        smoke_test=smoke_test,
        record_path=record,
        failed_record_path=failed_record,
    )
    if job_record.exit_status != "completed":
        raise typer.Exit(_wrapper_exit_code(job_record.exit_code))


def job_identity(command: str) -> JobIdentity:
    """
    Return the identity a job's wrapper command will record.

    `command` is the rendered wrapper prefix of a rule, ending with `--`.
    It is parsed with the wrapper's own options, so a planned job gets
    the identity its record will have.
    """
    arguments = shlex.split(command)
    prefix = ["python", "-m", "afabench.core.job_record"]
    if arguments[:3] != prefix or arguments[-1] != "--":
        message = f"Not a job record wrapper command: {command!r}"
        raise ValueError(message)
    # Resilient parsing: the script command after -- is not rendered yet.
    options = (
        typer.main.get_command(app)
        .make_context("job_record", arguments[3:], resilient_parsing=True)
        .params
    )
    return JobIdentity(
        **{field.name: options[field.name] for field in fields(JobIdentity)}
    )


def _wrapper_exit_code(script_exit_code: int | None) -> int:
    """Propagate the script's exit code, as a shell would for a signal."""
    if script_exit_code is None or script_exit_code == 0:
        # The script exited cleanly after SIGTERM, but the job timed out.
        return 128 + signal.SIGTERM
    if script_exit_code < 0:
        return 128 - script_exit_code
    return script_exit_code


if __name__ == "__main__":
    app()
