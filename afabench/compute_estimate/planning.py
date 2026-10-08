"""
Plan the jobs a Snakemake invocation would run, with their allocations.

The plan is the first half of a compute estimate
(`docs/adr/0006-job-records-beside-artifacts.md`). It comes from
Snakemake's own command-line handling, so profiles, workflow profiles,
`--config` and `--set-resources` resolve exactly as in a real run, and an
invalid invocation fails with the same error. Only the scheduling step is
replaced: the DAG is built and every job's resources and params are
evaluated, but nothing is submitted or run.

Needs Snakemake, a development dependency that `uv sync` installs.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from unittest import mock

from snakemake.cli import args_to_api, parse_args
from snakemake.common import async_run
from snakemake.exceptions import print_exception
from snakemake.jobs import Job
from snakemake.workflow import Workflow
from snakemake_interface_executor_plugins.registry import Plugin

type Device = Literal["cpu", "cuda"]


@dataclass(frozen=True, kw_only=True)
class PlannedJob:
    """A job an invocation would run, and the allocation it would request."""

    rule: str
    wildcards: dict[str, str]
    device: Device
    cpus: int
    gpus: int


class InvocationError(Exception):
    """Snakemake rejected the invocation; it printed the reason."""


def plan_jobs(arguments: Sequence[str]) -> list[PlannedJob]:
    """Return the jobs `snakemake <arguments>` would run."""
    planned: list[PlannedJob] = []

    def plan(workflow: Workflow, executor_plugin: Plugin, **_: object) -> None:
        # The first steps of Workflow.execute, which has no public way to
        # stop before scheduling. Without taking the lock, so a running
        # invocation of the same workflow can be estimated.
        workflow._executor_plugin = executor_plugin  # noqa: SLF001
        assert workflow.dag_settings is not None
        assert workflow.execution_settings is not None
        workflow._prepare_dag(  # noqa: SLF001
            forceall=workflow.dag_settings.forceall,
            ignore_incomplete=workflow.execution_settings.ignore_incomplete,
            lock_warn_only=True,
            nolock=True,
            shadow_prefix=workflow.execution_settings.shadow_prefix,
        )
        workflow._build_dag()  # noqa: SLF001
        dag = workflow.dag
        assert dag is not None
        async_run(dag.postprocess(update_needrun=False, check_initial=True))
        planned.extend(
            _planned_job(job)
            for job in dag.needrun_jobs()
            # Rules without an action, such as target rules, never run.
            if not job.is_norun
        )

    try:
        parser, args = parse_args(list(arguments))
    except Exception as error:
        print_exception(error)
        raise InvocationError from error
    with mock.patch.object(Workflow, "execute", plan):
        if not args_to_api(args, parser):
            raise InvocationError
    return sorted(
        planned, key=lambda job: (job.rule, sorted(job.wildcards.items()))
    )


def _planned_job(job: Job) -> PlannedJob:
    # Evaluating params runs the workflow's allocation checks, as
    # submission does.
    job.params  # noqa: B018
    resources = job.resources
    gpus = _gpus(resources.get("gpu"), resources.get("gres"))
    return PlannedJob(
        rule=job.rule.name,
        wildcards={
            name: str(value)
            for name, value in (job.wildcards_dict or {}).items()
        },
        device="cuda" if gpus else "cpu",
        cpus=_cpus(resources.get("cpus_per_task"), job.threads),
        gpus=gpus,
    )


def _cpus(cpus_per_task: object, threads: int) -> int:
    # As the SLURM executor plugin requests them: cpus_per_task, at least 1,
    # when set, otherwise the job's threads.
    if isinstance(cpus_per_task, int) and cpus_per_task:
        return max(1, cpus_per_task)
    return threads


def _gpus(gpu: object, gres: object) -> int:
    if isinstance(gpu, int) and gpu > 0:
        return gpu
    if isinstance(gres, str):
        match = re.fullmatch(r"gpu(?::\w+)?:(\d+)", gres)
        if match:
            return int(match[1])
    return 0
