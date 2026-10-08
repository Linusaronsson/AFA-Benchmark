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

import shlex
from collections.abc import Callable, Iterable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Protocol

from execution import ExecutionPolicy

from afabench.core.job_record import (
    JobIdentity,
    Stage,
    allocated_cpus,
    allocated_gpus,
)
from afabench.core.output_layout import pretrain_seed_in_folder

OUTPUT_ROOT = Path("extra/output")
FAILED_JOB_RECORDS = OUTPUT_ROOT / "failed_job_records"

IDENTITY_FIELDS = {field.name for field in fields(JobIdentity)}
# A wildcard named after an identity field holds that field; these hold one
# under another name.
RENAMED_WILDCARDS = {"dataset": "dataset_key"}
# Identity fields no wildcard holds, passed by the rule
RULE_FIELDS = {"name", "train_seed", "eval_batch_size"}


class Wildcards(Protocol):
    def items(self) -> Iterable[tuple[str, str]]: ...


class Output(Protocol):
    job_record: str


class JobRecordCommands:
    """Render the wrapper command param of each computational rule."""

    def __init__(
        self, execution: ExecutionPolicy, *, smoke_test: bool
    ) -> None:
        self.execution: ExecutionPolicy = execution
        self.smoke_test: bool = smoke_test

    def param(
        self,
        stage: Stage,
        identity: Callable[[Wildcards], str | None],
        **fields: Callable[[Wildcards], object],
    ) -> Callable[[Wildcards, Output, Mapping[str, object], int], str]:
        """
        Return a rule's `job_record` param, rendered per job.

        `identity` is the execution policy identity the rule's allocation
        resources use; each of `fields` returns a job record field the
        wildcards do not hold. Snakemake does not track params that take
        resources or threads, so another allocation reruns no job.
        """

        def job_record(
            wildcards: Wildcards,
            output: Output,
            resources: Mapping[str, object],
            threads: int,
        ) -> str:
            return job_record_command(
                output.job_record,
                stage=stage,
                wildcards=wildcards,
                # Also checks the job's final allocation, for rules whose
                # script takes no device.
                device=self.execution.checked_device(
                    stage, identity(wildcards), resources
                ),
                resources=resources,
                threads=threads,
                smoke_test=self.smoke_test,
                **{name: field(wildcards) for name, field in fields.items()},
            )

        return job_record


def job_record_command(
    record: str,
    *,
    stage: str,
    wildcards: Wildcards,
    device: str,
    resources: Mapping[str, object],
    threads: int,
    smoke_test: bool,
    **fields: object,
) -> str:
    """Return the wrapper command to prefix a rule's script command with."""
    unknown = fields.keys() - RULE_FIELDS
    if unknown:
        message = f"Unknown job record fields: {sorted(unknown)}"
        raise ValueError(message)
    identity: dict[str, object] = {
        RENAMED_WILDCARDS.get(wildcard, wildcard): value
        for wildcard, value in wildcards.items()
        if RENAMED_WILDCARDS.get(wildcard, wildcard) in IDENTITY_FIELDS
    }
    pretrain_folder = dict(wildcards.items()).get("pretrain_folder")
    if pretrain_folder is not None:
        identity["pretrain_seed"] = pretrain_seed_in_folder(pretrain_folder)
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
