"""
Plan a pipeline invocation: the jobs its Snakemake graph schedules.

A pipeline plan is plain data, free of Snakemake's objects: the first half
of a compute estimate (`docs/adr/0007-job-records-beside-artifacts.md`),
and what any other inspection of an invocation reads. Callers resolve the
pipeline launcher's arguments with `workflow.run.snakemake_arguments`. A
config invocation of a downloaded release is planned as its preview
(`afabench.release.config_run`): released outputs are reused on the
condition that regenerated datasets pass the release's content checks.

The plan comes from Snakemake's own command-line handling, so profiles,
workflow profiles, `--config` and `--set-resources` resolve exactly as in a
real run, and an invalid invocation fails with the same error. Only the
scheduling step is replaced: the DAG is built and every scheduled job's
resources and params are evaluated, but nothing is submitted or run, and
the workflow's onstart hook, which records the workflow config, never runs.

Needs Snakemake, a development dependency that `uv sync` installs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from unittest import mock

from snakemake.cli import args_to_api, parse_args
from snakemake.common import async_run
from snakemake.exceptions import print_exception
from snakemake.jobs import Job
from snakemake.workflow import Workflow
from snakemake_interface_executor_plugins.registry import Plugin

from afabench.core.job_record import (
    Device,
    JobIdentity,
    JobType,
    allocated_cpus,
    allocated_gpus,
    job_identity,
)
from afabench.release.config_run import config_run_commands


@dataclass(frozen=True, kw_only=True)
class PlannedJob:
    """A job an invocation would run, and the allocation it would request."""

    rule: str
    wildcards: dict[str, str]
    # What the job's record will say it computed; None for rules that
    # write no job record (aggregation and visualization).
    identity: JobIdentity | None
    device: Device
    # None when the cluster's default applies.
    cpus: int | None
    gpus: int
    # Snakemake's id of the job in this plan
    jobid: int
    # The scheduled jobs whose outputs it needs; jobs whose outputs exist
    # are not scheduled.
    dependencies: frozenset[int] = frozenset()
    # Requested walltime, not the measured duration used for the estimate.
    time_limit_minutes: int | None = None

    @property
    def job_type(self) -> JobType | None:
        return None if self.identity is None else self.identity.job_type


@dataclass(frozen=True, kw_only=True)
class PipelinePlan:
    """What a pipeline invocation would do, without Snakemake's objects."""

    # The jobs it would run, sorted by rule and wildcards. Rules without an
    # action, such as target rules, never run and are not among them.
    scheduled_jobs: list[PlannedJob]


class InvocationError(Exception):
    """Snakemake rejected the invocation; it printed the reason."""


def plan_invocation(snakemake_arguments: Sequence[str]) -> PipelinePlan:
    """Plan `snakemake <snakemake_arguments>` as the launcher would run it."""
    arguments = config_run_commands(snakemake_arguments, preview=True)[0]
    return PipelinePlan(scheduled_jobs=_scheduled_jobs(arguments))


def _scheduled_jobs(arguments: Sequence[str]) -> list[PlannedJob]:
    """Return the jobs `snakemake <arguments>` would run."""
    planned: list[PlannedJob] = []

    def plan(workflow: Workflow, executor_plugin: Plugin, **_: object) -> None:
        # The first steps of Workflow.execute, which has no public way to
        # stop before scheduling. Without taking the lock, so a running
        # invocation of the same workflow can be planned.
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
        jobs = [
            job
            for job in dag.needrun_jobs()
            # Rules without an action, such as target rules, never run.
            if not job.is_norun
        ]
        jobids = {job: job.jobid for job in jobs}
        planned.extend(
            _planned_job(
                job,
                jobid=jobids[job],
                dependencies=frozenset(
                    jobids[dependency]
                    for dependency in dag.dependencies[job]
                    if dependency in jobids
                ),
            )
            for job in jobs
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


def _planned_job(
    job: Job, *, jobid: int, dependencies: frozenset[int]
) -> PlannedJob:
    # As at submission, release the initial deferral of input-dependent
    # resources. Missing input files can still resolve to <TBD> in a plan.
    job.reset_params_and_resources()
    # Evaluating params runs the workflow's allocation checks, as
    # submission does.
    params = job.params
    assert params is not None
    job_record_command = params.get("job_record")
    resources = job.resources
    gpus = allocated_gpus(resources)
    runtime = resources.get("runtime")
    return PlannedJob(
        rule=job.rule.name,
        wildcards={
            name: str(value)
            for name, value in (job.wildcards_dict or {}).items()
        },
        identity=None
        if job_record_command is None
        else job_identity(job_record_command),
        device="cuda" if gpus else "cpu",
        cpus=allocated_cpus(resources, job.threads),
        gpus=gpus,
        jobid=jobid,
        dependencies=dependencies,
        time_limit_minutes=runtime if isinstance(runtime, int) else None,
    )
