"""
What the interactive preview shows of a pipeline plan, apart from widgets.

The scheduled jobs form a tree of groups, stage → method → dataset, with the
individual jobs below the last level. A group's summary and a job's details
are plain text lines. The dimensions come from each job's identity, what
its job record will say; a job without one, such as an aggregation or
visualization job, falls under explicit "no method" and "no dataset"
groups rather than an identity read off its wildcards.
"""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.planning import PipelinePlan, PlannedJob

type Dimension = Literal["stage", "method", "dataset"]
GROUPING: tuple[Dimension, ...] = ("stage", "method", "dataset")
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
# The stages whose job record names a method; a classifier training or
# pretraining job's name is a classifier or pretrained model, shared by
# methods.
METHOD_STAGES = {"training", "evaluation", "transformation"}
NOT_APPLICABLE = "not applicable"
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
    """One scheduled job below the last grouping level."""

    label: str
    job: PlannedJob


@dataclass(frozen=True)
class GroupNode:
    """The jobs sharing one value of a dimension, under their parent group."""

    dimension: Dimension
    # None: the dimension does not apply to the group's jobs.
    value: str | None
    children: tuple["GroupNode | JobNode", ...]

    @property
    def label(self) -> str:
        return f"no {self.dimension}" if self.value is None else self.value

    @property
    def job_count(self) -> int:
        return sum(
            child.job_count if isinstance(child, GroupNode) else 1
            for child in self.children
        )

    @property
    def jobs(self) -> list[PlannedJob]:
        return [
            job
            for child in self.children
            for job in (
                child.jobs if isinstance(child, GroupNode) else [child.job]
            )
        ]


def preview_tree(plan: PipelinePlan) -> tuple[GroupNode, ...]:
    """Return the plan's scheduled jobs grouped stage → method → dataset."""
    return _grouped(plan.scheduled_jobs, GROUPING)


def _grouped(
    jobs: Iterable[PlannedJob], grouping: tuple[Dimension, ...]
) -> tuple[GroupNode, ...]:
    if not grouping:
        return ()
    dimension, below = grouping[0], grouping[1:]
    groups: defaultdict[str | None, list[PlannedJob]] = defaultdict(list)
    for job in jobs:
        groups[dimension_of(job, dimension)].append(job)
    return tuple(
        GroupNode(
            dimension=dimension,
            value=value,
            children=_grouped(members, below)
            if below
            else tuple(
                JobNode(label=job_label(job), job=job)
                for job in sorted(members, key=job_label)
            ),
        )
        for value, members in sorted(
            groups.items(), key=lambda item: _group_order(dimension, item[0])
        )
    )


def _group_order(dimension: Dimension, value: str | None) -> tuple[int, str]:
    """Order stages as the pipeline runs them; other values alphabetically."""
    if value is None:
        return (len(STAGES) + 1, "")
    if dimension == "stage":
        return (STAGES.index(value) if value in STAGES else len(STAGES), value)
    return (0, value)


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
    """Describe the group: its dimension values and its jobs by subgroup."""
    lines = [f"{group.dimension.capitalize()}: {group.label}"]
    count = group.job_count
    lines.append(f"{count} scheduled job{'' if count == 1 else 's'}")
    subgroups = [
        child for child in group.children if isinstance(child, GroupNode)
    ]
    if subgroups:
        lines.append(
            f"By {subgroups[0].dimension}: "
            + ", ".join(f"{sub.label} ({sub.job_count})" for sub in subgroups)
        )
    return "\n".join(lines)


def job_details(job: PlannedJob, plan: PipelinePlan) -> list[tuple[str, str]]:
    """Return the job's rule, identity, allocation and prerequisites."""
    details = [
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
        ("Needs", _needs(job, plan)),
    ]
    return details


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


def _needs(job: PlannedJob, plan: PipelinePlan) -> str:
    """Name the scheduled jobs whose outputs the job needs."""
    labels = sorted(
        job_label(needed)
        for needed in plan.scheduled_jobs
        if needed.jobid in job.dependencies
    )
    return "\n".join(labels) if labels else "nothing scheduled"
