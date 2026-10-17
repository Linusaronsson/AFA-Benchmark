"""
What the interactive preview shows of a pipeline plan, apart from widgets.

The plan's jobs, run and reused, form a tree of groups, stage → method →
dataset by default or any ordered subset of those dimensions, with the
individual jobs below the last level; every job is in one place and counts
once. A view (`PlanView`) chooses the grouping and filters and searches the
jobs, all on the loaded plan. Each group counts its jobs by status, and a
job's details give its status with Snakemake's reasons for it, its
identity, paths and allocation; its relations are the jobs it needs and
the jobs needing it, whether the view shows them or not, and its shell
command is apart. The dimensions come from each job's identity, what its
job record will say. A shared prerequisite's job record names a classifier
or pretrained model, not a method, so those jobs are grouped as "Shared
prerequisites"; a job without a job record, such as an aggregation or
visualization job, is grouped as "Across methods" and "Across datasets".
A release preview's conditionally reused jobs come with a warning that
execution checks the regenerated datasets they wait for.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import permutations
from typing import Literal

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.planning import (
    JOB_STATUSES,
    JobStatus,
    PipelinePlan,
    PlannedJob,
    ReasonKind,
    StatusReason,
)

type Dimension = Literal["stage", "method", "dataset"]
DIMENSIONS: tuple[Dimension, ...] = ("stage", "method", "dataset")
# The default grouping
GROUPING: tuple[Dimension, ...] = DIMENSIONS
# Every grouping: each ordered subset of the dimensions, the default first
GROUPINGS: tuple[tuple[Dimension, ...], ...] = tuple(
    grouping
    for size in (3, 2, 1)
    for grouping in permutations(DIMENSIONS, size)
)
# The pipeline stages (CONTEXT.md) in pipeline order, as displayed.
STAGES = (
    "dataset generation",
    "classifier training",
    "pretraining",
    "training",
    "evaluation",
    "transformation",
    "aggregation",
    "visualization",
)
# The stage of each rule that writes no job record.
STAGE_OF_RULE = {
    "merge_eval_perf": "aggregation",
    "split_by_classifier_type": "aggregation",
    "collect_job_records": "aggregation",
    "plot_eval_perf": "visualization",
    "plot_eval_perf_per_dataset": "visualization",
    "plot_eval_actions": "visualization",
    "plot_time": "visualization",
}
# The stages whose job record names a method; a dataset generation,
# classifier training or pretraining job's name is a dataset, classifier
# or pretrained model: a shared prerequisite of methods.
METHOD_STAGES = {"training", "evaluation", "transformation"}
# The groups of jobs a dimension does not apply to, in their order after
# the groups of values
SHARED_PREREQUISITES = "Shared prerequisites"
ACROSS_METHODS = "Across methods"
ACROSS_DATASETS = "Across datasets"
EXPLICIT_GROUPS = (SHARED_PREREQUISITES, ACROSS_METHODS, ACROSS_DATASETS)
NO_STAGE = "no stage"
NOT_APPLICABLE = "not applicable"
# What a shared prerequisite's job record names, by stage: not a method
NAME_LABELS = {
    "classifier_training": "Classifier",
    "pretraining": "Pretrained model",
}
OUTSIDE_FILTERED_TREE = (
    "Outside the filtered tree: the filters or search hide this job. They, "
    "the tree and its counts are unchanged."
)
NO_SHELL_COMMAND = (
    "No shell command: the rule has none, or Snakemake could not format it "
    "without running the job. The preview runs nothing to obtain one."
)
# The text badge of each status, beside its color
STATUS_LABELS: dict[JobStatus, str] = {
    "run": "Run",
    "reuse": "Reuse",
    "conditional_reuse": "Conditional reuse",
}
# Only a release preview reuses conditionally; other plans' summaries leave
# the status out.
RELEASE_STATUSES: frozenset[JobStatus] = frozenset({"conditional_reuse"})
RELEASE_WARNING = (
    "Conditional reuse: regenerated datasets must match the release's "
    "recorded content before released outputs are reused during execution. "
    "Those datasets do not exist yet, so the preview has not checked them."
)
REASON_TEXTS: dict[ReasonKind, str] = {
    "missing_output": "Missing outputs",
    "incomplete_output": "Incomplete outputs",
    "updated_input": "Inputs newer than its outputs",
    "input_from_rerun": "Inputs another job of this plan rewrites",
    "input_set_changed": "The set of inputs changed since the last run",
    "code_changed": "The code changed since the last run",
    "params_changed": "Params changed since the last run",
    "software_changed": "The software environment changed since the last run",
    "forced": "Forced by the invocation",
    "no_output": "The rule has no output, so it always runs",
    "outputs_up_to_date": (
        "The outputs it is needed for exist and are up to date, and nothing "
        "it needs is rerun"
    ),
    "pending_dataset_check": (
        "Released outputs, reused during execution only if these regenerated "
        "dataset bundles match the release's recorded content"
    ),
    "conditional_input": (
        "Reads outputs that are themselves conditionally reused"
    ),
}
# A job's identity fields beyond its group's dimensions, with the short
# label of its tree entry and the label of its details.
IDENTITY_LABELS: tuple[tuple[str, str, str], ...] = (
    ("dataset_realization_index", "realization", "Dataset realization"),
    ("pretrain_seed", "pretrain seed", "Pretraining seed"),
    ("train_seed", "train seed", "Training seed"),
    ("eval_seed", "eval seed", "Evaluation seed"),
    ("train_hard_budget", "train hard budget", "Training hard budget"),
    (
        "train_soft_budget_param",
        "train soft-budget parameter",
        "Training soft-budget parameter",
    ),
    ("eval_hard_budget", "eval hard budget", "Evaluation hard budget"),
    (
        "eval_soft_budget_param",
        "eval soft-budget parameter",
        "Evaluation soft-budget parameter",
    ),
    ("eval_batch_size", "eval batch size", "Evaluation batch size"),
)
# The batch size changes no result, so it does not tell jobs apart.
LABEL_FIELDS = tuple(
    field for field, _, _ in IDENTITY_LABELS if field != "eval_batch_size"
)


@dataclass(frozen=True)
class JobNode:
    """One job below the last grouping level."""

    label: str
    job: PlannedJob

    @property
    def status(self) -> JobStatus:
        return self.job.status


@dataclass(frozen=True)
class GroupNode:
    """The jobs sharing one value of a dimension, under their parent group."""

    dimension: Dimension
    # The dimension's value, or the explicit group of jobs it does not
    # apply to.
    label: str
    children: tuple["GroupNode | JobNode", ...]

    @property
    def jobs(self) -> list[PlannedJob]:
        return [
            job
            for child in self.children
            for job in (
                child.jobs if isinstance(child, GroupNode) else [child.job]
            )
        ]

    @property
    def job_count(self) -> int:
        return len(self.jobs)

    @property
    def status_counts(self) -> dict[JobStatus, int]:
        return status_counts(self.jobs)

    @property
    def counts_label(self) -> str:
        """The group's jobs by status, leaving out statuses it has none of."""
        return counts_text(self.status_counts, with_zero=False)


def status_counts(jobs: Iterable[PlannedJob]) -> dict[JobStatus, int]:
    counted = Counter(job.status for job in jobs)
    return {status: counted[status] for status in JOB_STATUSES}


def counts_text(counts: dict[JobStatus, int], *, with_zero: bool) -> str:
    return ", ".join(
        f"{count} {STATUS_LABELS[status]}"
        for status, count in counts.items()
        if count or (with_zero and status not in RELEASE_STATUSES)
    )


def plan_summary(plan: PipelinePlan) -> str:
    """Count the plan's jobs, once each, by status."""
    count = len(plan.jobs)
    return f"{count} job{'' if count == 1 else 's'}: " + counts_text(
        status_counts(plan.jobs), with_zero=True
    )


def release_warning(plan: PipelinePlan) -> str | None:
    """Warn of the dataset checks a conditional reuse waits for, if any."""
    if any(job.status in RELEASE_STATUSES for job in plan.jobs):
        return RELEASE_WARNING
    return None


class GroupingError(ValueError):
    """A grouping that names no dimension, an unknown one or one twice."""


def parse_grouping(text: str) -> tuple[Dimension, ...]:
    """Return the grouping comma-separated dimensions name, in their order."""
    names = text.split(",")
    for name in names:
        if name not in DIMENSIONS:
            message = f"Unknown grouping dimension {name!r} in {text!r}; choose from {', '.join(DIMENSIONS)}"
            raise GroupingError(message)
    for name in names:
        if names.count(name) > 1:
            message = f"Grouping dimension {name!r} repeats in {text!r}"
            raise GroupingError(message)
    return tuple(DIMENSIONS[DIMENSIONS.index(name)] for name in names)


@dataclass(frozen=True, kw_only=True)
class PlanView:
    """
    How the preview shows a plan: its grouping, filters and search.

    A filter of None shows every value. The stage, dataset and status
    filters are strict; the method filter also shows the jobs its
    method's jobs need, directly or not, as far as the other filters do.
    """

    grouping: tuple[Dimension, ...] = GROUPING
    stage: str | None = None
    method: str | None = None
    dataset: str | None = None
    status: JobStatus | None = None
    # Whitespace-separated terms a shown job's rule, groups or label all
    # contain, regardless of case
    search: str = ""


# Every job in the default grouping
DEFAULT_VIEW = PlanView()


def preview_tree(
    plan: PipelinePlan, view: PlanView = DEFAULT_VIEW
) -> tuple[GroupNode, ...]:
    """Return the jobs the view shows of the plan, in its grouping."""
    return _grouped(shown_jobs(plan, view), view.grouping)


def shown_jobs(plan: PipelinePlan, view: PlanView) -> list[PlannedJob]:
    """Return the jobs of the plan the view's filters and search show."""
    jobs = (
        plan.jobs
        if view.method is None
        else _with_prerequisites(plan, view.method)
    )
    terms = view.search.lower().split()
    return [
        job
        for job in jobs
        if (view.stage is None or dimension_of(job, "stage") == view.stage)
        and (
            view.dataset is None
            or dimension_of(job, "dataset") == view.dataset
        )
        and (view.status is None or job.status == view.status)
        and all(term in _searched_text(job) for term in terms)
    ]


def _with_prerequisites(plan: PipelinePlan, method: str) -> list[PlannedJob]:
    """Return the method's jobs and the jobs they need, directly or not."""
    by_id = {job.jobid: job for job in plan.jobs}
    pending = [
        job.jobid for job in plan.jobs if dimension_of(job, "method") == method
    ]
    kept: set[int] = set()
    while pending:
        jobid = pending.pop()
        if jobid not in kept:
            kept.add(jobid)
            pending.extend(by_id[jobid].dependencies)
    return [job for job in plan.jobs if job.jobid in kept]


def _searched_text(job: PlannedJob) -> str:
    return " ".join(
        [job.rule, job_label(job)]
        + [group_of(job, dimension) for dimension in DIMENSIONS]
    ).lower()


def filter_values(plan: PipelinePlan, dimension: Dimension) -> list[str]:
    """Return the values of the dimension the plan's jobs have, in order."""
    values = {
        value
        for job in plan.jobs
        if (value := dimension_of(job, dimension)) is not None
    }
    return sorted(values, key=lambda value: _group_order(dimension, value))


def plan_statuses(plan: PipelinePlan) -> list[JobStatus]:
    """Return the statuses the plan's jobs have, in their order."""
    present = {job.status for job in plan.jobs}
    return [status for status in JOB_STATUSES if status in present]


def view_summary(plan: PipelinePlan, view: PlanView) -> str:
    """Count the jobs the view shows, beside the whole plan's when fewer."""
    jobs = shown_jobs(plan, view)
    if len(jobs) == len(plan.jobs):
        return plan_summary(plan)
    shown = (
        f"{len(jobs)} of {len(plan.jobs)} jobs: "
        + counts_text(status_counts(jobs), with_zero=False)
        if jobs
        else "no job matches the filters and search"
    )
    return f"Shown: {shown} | Whole plan: {plan_summary(plan)}"


def _grouped(
    jobs: Iterable[PlannedJob], grouping: tuple[Dimension, ...]
) -> tuple[GroupNode, ...]:
    if not grouping:
        return ()
    dimension, below = grouping[0], grouping[1:]
    groups: defaultdict[str, list[PlannedJob]] = defaultdict(list)
    for job in jobs:
        groups[group_of(job, dimension)].append(job)
    return tuple(
        GroupNode(
            dimension=dimension,
            label=label,
            children=_grouped(members, below)
            if below
            else tuple(
                JobNode(label=job_label(job), job=job)
                for job in sorted(members, key=job_label)
            ),
        )
        for label, members in sorted(
            groups.items(), key=lambda item: _group_order(dimension, item[0])
        )
    )


def _group_order(dimension: Dimension, label: str) -> tuple[int, str]:
    """Order stages as the pipeline runs them; other values alphabetically."""
    if label in EXPLICIT_GROUPS:
        return (len(STAGES) + 1 + EXPLICIT_GROUPS.index(label), label)
    if dimension == "stage":
        return (STAGES.index(label) if label in STAGES else len(STAGES), label)
    return (0, label)


def dimension_of(job: PlannedJob, dimension: Dimension) -> str | None:
    """Return the job's value of the dimension, None when it has none."""
    identity = job.identity
    if dimension == "stage":
        if identity is not None:
            return identity.stage.replace("_", " ")
        return STAGE_OF_RULE.get(job.rule)
    if identity is None:
        return None
    if dimension == "method":
        return identity.name if identity.stage in METHOD_STAGES else None
    return identity.dataset_key


def group_of(job: PlannedJob, dimension: Dimension) -> str:
    """Return the job's group of the dimension: its value or an explicit group."""
    value = dimension_of(job, dimension)
    if value is not None:
        return value
    if dimension == "stage":
        return NO_STAGE
    if dimension == "dataset":
        return ACROSS_DATASETS
    return ACROSS_METHODS if job.identity is None else SHARED_PREREQUISITES


def job_label(job: PlannedJob) -> str:
    """Name the job by its rule and what tells it from its group's jobs."""
    if job.identity is None:
        parts = [f"{name}={value}" for name, value in job.wildcards.items()]
    else:
        parts = [
            f"{short} {value}"
            for field, short, _ in IDENTITY_LABELS
            if field in LABEL_FIELDS
            and (value := getattr(job.identity, field)) is not None
        ]
    return f"{job.rule}: {', '.join(parts)}" if parts else job.rule


def group_summary(group: GroupNode) -> str:
    """Describe the group: its dimension value and its jobs by status and subgroup."""
    count = group.job_count
    lines = [
        f"{group.dimension.capitalize()}: {group.label}",
        f"{count} job{'' if count == 1 else 's'}: "
        + counts_text(group.status_counts, with_zero=False),
    ]
    subgroups = [
        child for child in group.children if isinstance(child, GroupNode)
    ]
    if subgroups:
        lines.append(
            f"By {subgroups[0].dimension}: "
            + ", ".join(
                f"{sub.label} ({sub.counts_label})" for sub in subgroups
            )
        )
    return "\n".join(lines)


def job_details(job: PlannedJob, plan: PipelinePlan) -> list[tuple[str, str]]:
    """Return the job's status, identity, allocation, paths and existing inputs."""
    details = [
        ("Status", STATUS_LABELS[job.status]),
        ("Reason", "\n".join(map(reason_text, job.reasons)) or "none given"),
        ("Rule", job.rule),
        ("Stage", _shown(dimension_of(job, "stage"))),
    ]
    identity = job.identity
    if identity is None:
        details.append(("Job record", "none: the rule writes no job record"))
    details += [
        ("Method", _shown(dimension_of(job, "method"))),
        ("Dataset", _shown(dimension_of(job, "dataset"))),
    ]
    if identity is not None:
        if identity.stage in NAME_LABELS:
            details.append(
                (NAME_LABELS[identity.stage], _shown(identity.name))
            )
        details += _identity_details(identity)
    details += [
        ("Wildcards", _wildcards(job)),
        ("Device", job.device),
        ("CPUs", "cluster default" if job.cpus is None else str(job.cpus)),
        ("GPUs", str(job.gpus)),
        (
            "Time limit",
            "none requested"
            if job.time_limit_minutes is None
            else f"{job.time_limit_minutes} minutes",
        ),
        (
            "Placement",
            "\n".join(
                f"{name}={value}" for name, value in job.placement.items()
            )
            or "none requested",
        ),
        ("Inputs", _lines(job.inputs)),
        ("Outputs", _lines(job.outputs)),
        ("Existing inputs", _lines(existing_inputs(job, plan))),
    ]
    if group_of(job, "method") == SHARED_PREREQUISITES:
        details.append(
            ("Consuming methods", _lines(consuming_methods(job, plan), ", "))
        )
    return details


def reason_text(reason: StatusReason) -> str:
    """Put one of Snakemake's reasons in words."""
    text = REASON_TEXTS[reason.kind]
    if reason.detail is not None:
        text += f": {reason.detail}"
    if reason.files:
        text += ": " + ", ".join(reason.files)
    return text


def existing_inputs(job: PlannedJob, plan: PipelinePlan) -> list[str]:
    """Return the job's inputs no job of the plan makes, as a stage that is off writes."""
    made = {output for other in plan.jobs for output in other.outputs}
    return [path for path in job.inputs if path not in made]


def prerequisites(job: PlannedJob, plan: PipelinePlan) -> list[PlannedJob]:
    """Return the jobs of the plan whose outputs the job needs."""
    return [needed for needed in plan.jobs if needed.jobid in job.dependencies]


def consumers(job: PlannedJob, plan: PipelinePlan) -> list[PlannedJob]:
    """Return the jobs of the plan that need the job's outputs."""
    return [other for other in plan.jobs if job.jobid in other.dependencies]


def consuming_methods(job: PlannedJob, plan: PipelinePlan) -> list[str]:
    """Return the methods whose jobs need the job's outputs, directly or not."""
    methods: set[str] = set()
    pending = consumers(job, plan)
    seen: set[int] = set()
    while pending:
        consumer = pending.pop()
        if consumer.jobid in seen:
            continue
        seen.add(consumer.jobid)
        method = dimension_of(consumer, "method")
        if method is not None:
            methods.add(method)
        pending.extend(consumers(consumer, plan))
    return sorted(methods)


type Direction = Literal["needs", "needed_by"]


@dataclass(frozen=True)
class Relation:
    """A job another job needs or is needed by, and whether the view shows it."""

    direction: Direction
    job: PlannedJob
    shown: bool

    @property
    def title(self) -> str:
        return job_title(self.job)


def job_relations(
    job: PlannedJob, plan: PipelinePlan, view: PlanView
) -> list[Relation]:
    """Return the jobs the job needs, then the jobs needing it, by title."""
    shown = {shown.jobid for shown in shown_jobs(plan, view)}
    related: list[tuple[Direction, list[PlannedJob]]] = [
        ("needs", prerequisites(job, plan)),
        ("needed_by", consumers(job, plan)),
    ]
    return [
        Relation(direction=direction, job=other, shown=other.jobid in shown)
        for direction, others in related
        for other in sorted(others, key=job_title)
    ]


def job_title(job: PlannedJob) -> str:
    """Name the job apart from its groups: its label, then its names."""
    identity = job.identity
    names = (
        []
        if identity is None
        else list(
            dict.fromkeys(
                name
                for name in (identity.name, identity.dataset_key)
                if name is not None
            )
        )
    )
    label = job_label(job)
    return f"{label} ({', '.join(names)})" if names else label


def shell_command_text(job: PlannedJob) -> str:
    """Return the job's shell command, or why there is none."""
    return job.shell_command or NO_SHELL_COMMAND


def _identity_details(identity: JobIdentity) -> list[tuple[str, str]]:
    return [
        (label, _shown(getattr(identity, field)))
        for field, _, label in IDENTITY_LABELS
    ]


def _shown(value: object) -> str:
    return NOT_APPLICABLE if value is None else str(value)


def _wildcards(job: PlannedJob) -> str:
    return (
        ", ".join(f"{name}={value}" for name, value in job.wildcards.items())
        or "none"
    )


def _lines(items: Iterable[str], separator: str = "\n") -> str:
    return separator.join(items) or "none"
