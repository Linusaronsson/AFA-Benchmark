"""
Plan a pipeline invocation: the jobs of its Snakemake graph and their status.

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
scheduling step is replaced: the DAG is built and every job's resources
and params are evaluated, and its shell command formatted from them as a
dry run prints it, but nothing is submitted or run, and the workflow's
onstart hook, which records the workflow config, never runs. A job to run
whose evaluation fails fails planning, as its submission would fail the
run; a reused job, which a run never submits, is planned with its
identity, allocation and shell command unavailable instead.

The plan holds every job of the requested targets' dependency graph that
has an action, with the status Snakemake decided for it: a job it would
run, or a job whose existing outputs it reuses. Both decisions and their
reasons are Snakemake's own; the planner neither forces jobs nor reads
file existence itself. A file an input needs that no rule of the
invocation produces, such as the output of a stage that is off, is read
as it is and is no job of the plan.

One decision is the release's, not Snakemake's. A release preview plans
dataset generation and the jobs reading its datasets in one graph, so
Snakemake reruns every released job whose dataset bundles are omitted and
must be regenerated. Execution generates them first and reuses those jobs
if their content matches the release's record. A job the preview runs only
for such datasets, directly or through other such jobs, is planned as
conditionally reused; the release (`afabench.release.reuse`) names the
datasets whose check it left to execution.

Needs Snakemake, a development dependency that `uv sync` installs.
"""

import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Literal
from unittest import mock

from snakemake.cli import args_to_api, parse_args
from snakemake.common import async_run
from snakemake.exceptions import (
    RuleException,
    WorkflowError,
    print_exception,
)
from snakemake.jobs import Job, Reason
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
from afabench.release.reuse import ReleaseReuse

# What the invocation does with a job: run it, including to replace its
# existing but stale outputs, reuse its existing outputs without running
# it, or reuse its released outputs once the datasets it reads are
# regenerated and pass the release's content checks. Independent of
# whether its output files happen to exist.
type JobStatus = Literal["run", "reuse", "conditional_reuse"]
JOB_STATUSES: tuple[JobStatus, ...] = ("run", "reuse", "conditional_reuse")

# The resources the SLURM executor places a job with, beyond its CPUs, GPUs
# and time limit
PLACEMENT_RESOURCES = (
    "slurm_partition",
    "slurm_account",
    "clusters",
    "qos",
    "constraint",
    "gres",
    "gpu_model",
    "mem_mb",
    "mem_mb_per_cpu",
    "slurm_extra",
)

# Why a job has its status. "outputs_up_to_date" is Snakemake's one reason
# to reuse: the outputs its consumers need exist and are up to date, so an
# output nothing needs may still be missing. "pending_dataset_check" and
# "conditional_input" are the reasons to reuse conditionally: the
# regenerated datasets whose check the reuse waits for, and the outputs of
# conditionally reused jobs it reads. The others are Snakemake's reasons to
# run. The files a reason names are the job's missing outputs, its inputs
# newer than its outputs, or its inputs another job of the plan rewrites.
type ReasonKind = Literal[
    "missing_output",
    "incomplete_output",
    "updated_input",
    "input_from_rerun",
    "input_set_changed",
    "code_changed",
    "params_changed",
    "software_changed",
    "forced",
    "no_output",
    "outputs_up_to_date",
    "pending_dataset_check",
    "conditional_input",
]


@dataclass(frozen=True, kw_only=True)
class StatusReason:
    """One of Snakemake's reasons for a job's status."""

    kind: ReasonKind
    files: tuple[str, ...] = ()
    # What Snakemake adds, such as which params changed.
    detail: str | None = None


@dataclass(frozen=True, kw_only=True)
class PlannedJob:
    """A job of an invocation's graph, and the allocation it would request."""

    rule: str
    wildcards: dict[str, str]
    # What the job's record will say it computed; None for rules that
    # write no job record (aggregation and visualization).
    identity: JobIdentity | None
    # None when unavailable.
    device: Device | None
    # None when the cluster's default applies, or unavailable.
    cpus: int | None
    # None when unavailable.
    gpus: int | None
    # The placement resources it requests (PLACEMENT_RESOURCES), by name
    placement: dict[str, str] = field(default_factory=dict)
    # Snakemake's id of the job in this plan
    jobid: int
    status: JobStatus
    reasons: tuple[StatusReason, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    # The jobs of the plan whose outputs it needs, run or reused.
    dependencies: frozenset[int] = frozenset()
    # Requested walltime, not the measured duration used for the estimate.
    time_limit_minutes: int | None = None
    # The shell command it would run, without its indentation; None when
    # the rule has none or it cannot be formatted without running the job.
    shell_command: str | None = None
    # Why its identity, allocation and shell command are unavailable:
    # evaluating its params or resources as for submission failed. Only for
    # a reused job, which a run never submits; None otherwise.
    unavailable: str | None = None

    @property
    def job_type(self) -> JobType | None:
        return None if self.identity is None else self.identity.job_type


@dataclass(frozen=True, kw_only=True)
class PipelinePlan:
    """What a pipeline invocation would do, without Snakemake's objects."""

    # Every job of the requested targets' graph that has an action, run or
    # reused, sorted by rule and wildcards. Rules without an action, such as
    # target rules, never run and are not among them.
    jobs: list[PlannedJob]

    @property
    def scheduled_jobs(self) -> list[PlannedJob]:
        """
        The jobs it would run, depending on scheduled jobs only.

        Conditionally reused jobs count among them, as Snakemake schedules
        them in a release preview.
        """
        scheduled = {job.jobid for job in self.jobs if job.status != "reuse"}
        return [
            replace(job, dependencies=job.dependencies & scheduled)
            for job in self.jobs
            if job.jobid in scheduled
        ]


class InvocationError(Exception):
    """Snakemake rejected the invocation; it printed the reason."""


def plan_invocation(snakemake_arguments: Sequence[str]) -> PipelinePlan:
    """Plan `snakemake <snakemake_arguments>` as the launcher would run it."""
    arguments = config_run_commands(snakemake_arguments, preview=True)[0]
    return PipelinePlan(jobs=_graph_jobs(arguments))


def _graph_jobs(arguments: Sequence[str]) -> list[PlannedJob]:
    """Return the jobs of the graph `snakemake <arguments>` would run from."""
    planned: list[PlannedJob] = []
    pending_dataset_checks: dict[str, set[str]] = {}

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
        # Building the DAG decides which jobs need to run and why.
        workflow._build_dag()  # noqa: SLF001
        dag = workflow.dag
        assert dag is not None
        async_run(dag.postprocess(update_needrun=False, check_initial=True))
        jobs = [
            job
            for job in dag.jobs
            # Rules without an action, such as target rules, never run.
            if not job.is_norun
        ]
        jobids = {job: job.jobid for job in jobs}
        planned.extend(
            _evaluated_job(
                job,
                jobid=jobids[job],
                status="run" if dag.needrun(job) else "reuse",
                reasons=tuple(_reasons(dag.reason(job))),
                dependencies=frozenset(
                    jobids[dependency]
                    for dependency in dag.dependencies[job]
                    if dependency in jobids
                ),
            )
            for job in jobs
        )
        # The workflow's checked release reuse (workflow/snakefiles/settings.smk)
        release_reuse = workflow.globals.get("RELEASE_REUSE")
        if isinstance(release_reuse, ReleaseReuse):
            pending_dataset_checks.update(
                (str(consumer), {str(dataset) for dataset in datasets})
                for consumer, datasets in (
                    release_reuse.pending_dataset_checks.items()
                )
            )

    try:
        parser, args = parse_args(list(arguments))
    except Exception as error:
        print_exception(error)
        raise InvocationError from error
    with mock.patch.object(Workflow, "execute", plan):
        if not args_to_api(args, parser):
            raise InvocationError
    return _with_conditional_reuse(
        sorted(
            planned, key=lambda job: (job.rule, sorted(job.wildcards.items()))
        ),
        pending_dataset_checks,
    )


def _with_conditional_reuse(
    jobs: list[PlannedJob], pending_dataset_checks: Mapping[str, set[str]]
) -> list[PlannedJob]:
    """
    Plan as conditionally reused the jobs run only for regenerated datasets.

    `pending_dataset_checks` names, per released output, the datasets whose
    content check the release left to execution. A job is conditionally
    reused when Snakemake runs it only because jobs of the plan rewrite its
    inputs, and each of those inputs is such a dataset of its outputs or an
    output of a conditionally reused job.
    """
    if not pending_dataset_checks:
        return jobs
    by_id = {job.jobid: job for job in jobs}
    conditional_outputs: set[str] = set()
    graph = {job.jobid: job.dependencies for job in jobs}
    for jobid in TopologicalSorter(graph).static_order():
        job = by_id[jobid]
        reasons = _conditional_reasons(
            job, pending_dataset_checks, conditional_outputs
        )
        if reasons:
            by_id[jobid] = replace(
                job, status="conditional_reuse", reasons=reasons
            )
            conditional_outputs.update(map(_normalized, job.outputs))
    return [by_id[job.jobid] for job in jobs]


def _conditional_reasons(
    job: PlannedJob,
    pending_dataset_checks: Mapping[str, set[str]],
    conditional_outputs: set[str],
) -> tuple[StatusReason, ...]:
    """Return why the job is conditionally reused; none if it is not."""
    if job.status != "run" or any(
        reason.kind != "input_from_rerun" for reason in job.reasons
    ):
        return ()
    pending = {
        dataset
        for output in job.outputs
        for dataset in pending_dataset_checks.get(_normalized(output), ())
    }
    rewritten = [file for reason in job.reasons for file in reason.files]
    datasets = tuple(f for f in rewritten if _normalized(f) in pending)
    inputs = tuple(
        f for f in rewritten if _normalized(f) in conditional_outputs
    )
    if not rewritten or len(datasets) + len(inputs) != len(rewritten):
        return ()
    named: list[tuple[ReasonKind, tuple[str, ...]]] = [
        ("pending_dataset_check", datasets),
        ("conditional_input", inputs),
    ]
    return tuple(
        StatusReason(kind=kind, files=files) for kind, files in named if files
    )


def _normalized(path: str) -> str:
    """Spell a path as the release names it."""
    return str(Path(path))


def _evaluated_job(
    job: Job,
    *,
    jobid: int,
    status: JobStatus,
    reasons: tuple[StatusReason, ...],
    dependencies: frozenset[int],
) -> PlannedJob:
    """
    Plan the job with its evaluated params and resources.

    A job to run fails planning when they fail, as its submission would
    fail the run. A run never submits a reused job, so one whose evaluation
    fails is planned with its identity, allocation and shell command
    unavailable.
    """
    try:
        return _planned_job(
            job,
            jobid=jobid,
            status=status,
            reasons=reasons,
            dependencies=dependencies,
        )
    except WorkflowError as error:
        if status == "run":
            raise
        return PlannedJob(
            rule=job.rule.name,
            wildcards=_wildcards(job),
            identity=None,
            device=None,
            cpus=None,
            gpus=None,
            jobid=jobid,
            status=status,
            reasons=reasons,
            inputs=_paths(job.input),
            outputs=_paths(job.output),
            dependencies=dependencies,
            unavailable=str(error),
        )


def _planned_job(
    job: Job,
    *,
    jobid: int,
    status: JobStatus,
    reasons: tuple[StatusReason, ...],
    dependencies: frozenset[int],
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
        wildcards=_wildcards(job),
        identity=None
        if job_record_command is None
        else job_identity(job_record_command),
        device="cuda" if gpus else "cpu",
        cpus=allocated_cpus(resources, job.threads),
        gpus=gpus,
        placement={
            name: str(value)
            for name in PLACEMENT_RESOURCES
            if (value := resources.get(name))
        },
        jobid=jobid,
        status=status,
        reasons=reasons,
        inputs=_paths(job.input),
        outputs=_paths(job.output),
        dependencies=dependencies,
        time_limit_minutes=runtime if isinstance(runtime, int) else None,
        shell_command=_shell_command(job),
    )


def _shell_command(job: Job) -> str | None:
    """Format the job's shell command from its evaluated params, running nothing."""
    try:
        command = job.shellcmd
    except RuleException:
        return None
    return None if command is None else textwrap.dedent(command).strip()


def _wildcards(job: Job) -> dict[str, str]:
    return {
        name: str(value) for name, value in (job.wildcards_dict or {}).items()
    }


def _paths(files: Iterable[object]) -> tuple[str, ...]:
    return tuple(str(file) for file in files)


def _reasons(reason: Reason) -> Iterable[StatusReason]:
    """Translate Snakemake's reason to run a job; none means it is reused."""
    if not reason:
        yield StatusReason(kind="outputs_up_to_date")
    elif reason.forced:
        yield StatusReason(kind="forced")
    else:
        if reason.noio or reason.nooutput:
            yield StatusReason(kind="no_output")
        yield from _file_reasons(reason)
        yield from _change_reasons(reason)


def _file_reasons(reason: Reason) -> Iterable[StatusReason]:
    # Snakemake counts an input another job rewrites among the updated
    # inputs too; it is the rerun that explains it.
    # Snakemake's files are str subclasses.
    named: list[tuple[ReasonKind, set[str]]] = [
        ("missing_output", reason.missing_output),
        ("incomplete_output", reason.incomplete_output),
        ("updated_input", reason.updated_input - reason.updated_input_run),
        ("input_from_rerun", reason.updated_input_run),
    ]
    for kind, files in named:
        if files:
            yield StatusReason(kind=kind, files=_paths(sorted(files)))


def _change_reasons(reason: Reason) -> Iterable[StatusReason]:
    flagged: list[tuple[ReasonKind, bool]] = [
        ("input_set_changed", reason.input_changed),
        ("code_changed", reason.code_changed),
        ("software_changed", reason.software_stack_changed),
    ]
    for kind, changed in flagged:
        if changed:
            yield StatusReason(kind=kind)
    if reason.params_changed:
        yield StatusReason(
            kind="params_changed", detail=str(reason.params_changed)
        )
