"""
The job record wrapper command each computational rule runs its script with.

The wrapper (`afabench/core/job_record.py`) times the script and writes the
job record. This module renders its options from what Snakemake resolved
for the job: identity from the wildcards, allocation from the final
resources, after profile defaults and `--set-resources` overrides.

A failed or timed-out job's record goes to the same path under
`FAILED_JOB_RECORDS` instead, because Snakemake deletes a failed job's
declared outputs; it is not an output of any rule.
"""

import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Protocol

OUTPUT_ROOT = Path("extra/output")
FAILED_JOB_RECORDS = OUTPUT_ROOT / "failed_job_records"

# Wildcards whose value is the job record identity field of the same name
IDENTITY_WILDCARDS = {
    "dataset": "dataset_key",
    "dataset_realization_index": "dataset_realization_index",
    "pretrain_seed": "pretrain_seed",
    "train_seed": "train_seed",
    "eval_seed": "eval_seed",
    "train_hard_budget": "train_hard_budget",
    "train_soft_budget_param": "train_soft_budget_param",
    "eval_hard_budget": "eval_hard_budget",
    "eval_soft_budget_param": "eval_soft_budget_param",
}
# Identity fields no wildcard holds, passed by the rule
RULE_FIELDS = {"name", "train_seed", "eval_batch_size"}


class Wildcards(Protocol):
    def items(self) -> Iterable[tuple[str, str]]: ...


def job_record_command(
    record: str,
    *,
    stage: str,
    wildcards: Wildcards,
    device: str,
    resources: Mapping[str, object],
    threads: int,
    smoke_test: bool,
    time_file: str | None = None,
    **fields: object,
) -> str:
    """Return the wrapper command to prefix a rule's script command with."""
    unknown = fields.keys() - RULE_FIELDS
    if unknown:
        message = f"Unknown job record fields: {sorted(unknown)}"
        raise ValueError(message)
    identity: dict[str, object] = {
        IDENTITY_WILDCARDS[wildcard]: value
        for wildcard, value in wildcards.items()
        if wildcard in IDENTITY_WILDCARDS
    }
    pretrain_folder = dict(wildcards.items()).get("pretrain_folder", "")
    if pretrain_folder.startswith("pretrain_seed-"):
        identity["pretrain_seed"] = pretrain_folder.removeprefix(
            "pretrain_seed-"
        ).removesuffix("/")
    identity |= fields
    options: dict[str, object] = {
        "record": record,
        "failed_record": FAILED_JOB_RECORDS
        / Path(record).relative_to(OUTPUT_ROOT),
        "stage": stage,
        "device": device,
        "cpus": allocated_cpus(resources, threads),
        "gpus": allocated_gpus(resources),
        "time_limit_minutes": resources.get("runtime"),
        "time_file": time_file,
        **identity,
    }
    arguments = [
        "python",
        "-m",
        "afabench.core.job_record",
        "--smoke-test" if smoke_test else "--no-smoke-test",
    ]
    for option, value in options.items():
        # A "null" wildcard is a budget outside this job's setting.
        if value is not None and value != "null":
            arguments += [f"--{option.replace('_', '-')}", str(value)]
    return shlex.join([*arguments, "--"])


def allocated_cpus(
    resources: Mapping[str, object], threads: int
) -> int | None:
    """Return the CPUs the job requests, as the SLURM executor resolves them."""
    cpus_per_task = resources.get("cpus_per_task")
    if not cpus_per_task:
        return threads
    if not isinstance(cpus_per_task, int):
        message = f"cpus_per_task must be an integer, got {cpus_per_task!r}"
        raise TypeError(message)
    # A negative count leaves the CPUs to the cluster's default.
    return None if cpus_per_task < 0 else max(1, cpus_per_task)


def allocated_gpus(resources: Mapping[str, object]) -> int:
    """Return the GPUs the job requests through `gpu` or a GPU `gres`."""
    gpu = resources.get("gpu", 0)
    if gpu:
        return int(str(gpu))
    match = re.fullmatch(
        r"gpu(:[a-zA-Z0-9_]+)?:(\d+)", str(resources.get("gres", ""))
    )
    return int(match.group(2)) if match else 0
