"""
The W&B run of each job that logs metrics, which Snakemake chooses to link it.

A job's script starts its W&B run with the ID and name it finds in
WANDB_RUN_ID and WANDB_NAME, which the rule's `wandb` param sets in front of
its command. The name is the job's rule and wildcards. The ID describes the
job by its rule, name and dataset, and ends in what makes each attempt
unique: on a cluster the SLURM job ID, which the job reads from SLURM_JOB_ID
and the submitting Snakemake from the submission message; for a local job
random characters, chosen once by the one Snakemake process that runs it.
So a retried or rerun job gets another W&B run, and the link printed for a
job is the link to the run its script starts. A local job retried within one
invocation keeps its ID.

With use_wandb, WANDB_ENTITY and WANDB_PROJECT must be set: a link needs
both, and jobs inherit them, so their runs land where the links point.
"""

import random
import re
import shlex
import string
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol, Self

WANDB_VARIABLES = ("WANDB_ENTITY", "WANDB_PROJECT")
# The characters run IDs keep; any other becomes an underscore.
UNSAFE_ID_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]")
LOCAL_SUFFIX_LENGTH = 8


class Wildcards(Protocol):
    def items(self) -> Iterable[tuple[str, str]]: ...


# The job's name (method, pretrained model or classifier) from its wildcards
JobNameField = Callable[[Mapping[str, str]], str]


class WandbSettingsError(ValueError):
    """use_wandb without the environment variables a link needs."""


def job_name(rule: str, wildcards: Mapping[str, str]) -> str:
    """Name a job by its rule and wildcards, as the terminal does."""
    listed = ", ".join(f"{name}={value}" for name, value in wildcards.items())
    return f"{rule} ({listed})" if listed else rule


@dataclass(frozen=True)
class WandbProject:
    """The W&B entity and project every job logs to."""

    entity: str
    project: str

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> Self:
        missing = [
            name for name in WANDB_VARIABLES if not environment.get(name)
        ]
        if missing:
            message = f"use_wandb=true (workflow/run.py --wandb) needs {' and '.join(WANDB_VARIABLES)} set, so that each job's W&B run has a link; set {', '.join(missing)}"
            raise WandbSettingsError(message)
        return cls(
            entity=environment["WANDB_ENTITY"],
            project=environment["WANDB_PROJECT"],
        )

    def run_url(self, run_id: str) -> str:
        return f"https://wandb.ai/{self.entity}/{self.project}/runs/{run_id}"


class WandbRuns:
    """Choose, pass on and link the W&B run of each job that logs metrics."""

    def __init__(
        self, project: WandbProject | None, *, slurm_job_id: str | None
    ) -> None:
        """
        Name and link W&B runs in `project`, None without use_wandb.

        `slurm_job_id` is the SLURM job ID inside a job's own Snakemake on a
        cluster, None in the invocation's.
        """
        self.project: WandbProject | None = project
        self.slurm_job_id: str | None = slurm_job_id
        # Each rule whose script starts a W&B run, with its job name field
        self._names: dict[str, JobNameField] = {}
        self._local_suffixes: dict[
            tuple[str, tuple[tuple[str, str], ...]], str
        ] = {}
        self._rng: random.Random = random.Random()

    def param(
        self, rule: str, name: JobNameField
    ) -> Callable[[Wildcards, Mapping[str, object]], str]:
        """
        Return the `wandb` param of `rule`, whose script starts a W&B run.

        The param is empty or ends in a space, so a rule's shell command
        reads `{params.wandb}{params.image}...`. It takes the resources only
        because Snakemake does not track params that do: a new attempt's ID
        reruns no job.
        """
        self._names[rule] = name

        def wandb(
            wildcards: Wildcards,
            resources: Mapping[str, object],  # noqa: ARG001
        ) -> str:
            values = dict(wildcards.items())
            run_id = self.run_id(rule, values, self.slurm_job_id)
            if run_id is None:
                return ""
            return (
                f"WANDB_RUN_ID={shlex.quote(run_id)} "
                f"WANDB_NAME={shlex.quote(job_name(rule, values))} "
            )

        return wandb

    def run_id(
        self,
        rule: str,
        wildcards: Mapping[str, str],
        slurm_job_id: str | None,
    ) -> str | None:
        """
        Return the run ID of one attempt of a job, or None if it has no run.

        `slurm_job_id` is the attempt's SLURM job ID; without one, the job
        runs locally.
        """
        name = self._names.get(rule)
        if self.project is None or name is None:
            return None
        suffix = slurm_job_id or self._local_suffix(rule, wildcards)
        described = "-".join(
            [rule, name(wildcards), wildcards["dataset"], suffix]
        )
        return UNSAFE_ID_CHARACTERS.sub("_", described)

    def link(
        self,
        rule: str,
        wildcards: Mapping[str, str],
        slurm_job_id: str | None,
    ) -> str | None:
        """Return the URL of one attempt's W&B run, or None if it has none."""
        run_id = self.run_id(rule, wildcards, slurm_job_id)
        if self.project is None or run_id is None:
            return None
        return self.project.run_url(run_id)

    def _local_suffix(self, rule: str, wildcards: Mapping[str, str]) -> str:
        key = (rule, tuple(sorted(wildcards.items())))
        if key not in self._local_suffixes:
            self._local_suffixes[key] = "".join(
                self._rng.choices(
                    string.ascii_lowercase + string.digits,
                    k=LOCAL_SUFFIX_LENGTH,
                )
            )
        return self._local_suffixes[key]
