"""The preview's presentation of a pipeline plan, apart from its widgets (#158)."""

from dataclasses import replace

import pytest

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.planning import (
    JobStatus,
    PipelinePlan,
    PlannedJob,
    StatusReason,
)
from afabench.pipeline_preview.model import (
    DEFAULT_VIEW,
    GROUPINGS,
    Dimension,
    GroupNode,
    JobNode,
    PlanView,
    filter_values,
    group_summary,
    job_details,
    job_relations,
    parse_grouping,
    plan_statuses,
    plan_summary,
    preview_tree,
    release_warning,
    shell_command_text,
    shown_jobs,
    view_summary,
)


def job(
    rule: str,
    jobid: int,
    identity: JobIdentity | None,
    status: JobStatus = "run",
    reasons: tuple[StatusReason, ...] = (),
    wildcards: dict[str, str] | None = None,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    dependencies: frozenset[int] | None = None,
    cpus: int | None = 4,
    placement: dict[str, str] | None = None,
    shell_command: str | None = None,
) -> PlannedJob:
    return PlannedJob(
        rule=rule,
        wildcards=wildcards or {},
        identity=identity,
        device="cpu",
        cpus=cpus,
        gpus=0,
        jobid=jobid,
        status=status,
        reasons=reasons,
        inputs=inputs,
        outputs=outputs,
        dependencies=dependencies or frozenset(),
        time_limit_minutes=30,
        placement=placement or {},
        shell_command=shell_command,
    )


def training(
    jobid: int,
    method: str,
    realization: int,
    status: JobStatus = "run",
    hard_budget: int = 4,
) -> PlannedJob:
    return job(
        "train_method",
        jobid,
        JobIdentity(
            stage="training",
            name=method,
            dataset_key="cube",
            dataset_realization_index=realization,
            train_seed=0,
            train_hard_budget=hard_budget,
        ),
        status=status,
        reasons=(
            StatusReason(kind="outputs_up_to_date")
            if status == "reuse"
            else StatusReason(
                kind="missing_output",
                files=(f"{method}/{realization}.bundle",),
            ),
        ),
        inputs=("datasets/cube/0/train.bundle",),
        outputs=(f"{method}/{realization}.bundle",),
        dependencies=frozenset({20, 40}),
    )


EVALUATION = job(
    "eval_method",
    10,
    JobIdentity(
        stage="evaluation",
        name="alpha",
        dataset_key="cube",
        dataset_realization_index=0,
        train_seed=0,
        eval_seed=0,
        train_hard_budget=4,
        eval_hard_budget=2,
        eval_batch_size=32,
    ),
    reasons=(
        StatusReason(kind="input_from_rerun", files=("alpha/0.bundle",)),
        StatusReason(kind="params_changed", detail="eval_hard_budget"),
    ),
    inputs=("alpha/0.bundle", "datasets/cube/0/test.bundle"),
    outputs=("eval/alpha/0.csv",),
    dependencies=frozenset({1, 20}),
)
CLASSIFIER = job(
    "train_classifier",
    20,
    JobIdentity(
        stage="classifier_training",
        name="masked_mlp_classifier",
        dataset_key="cube",
        dataset_realization_index=0,
    ),
    status="reuse",
    reasons=(StatusReason(kind="outputs_up_to_date"),),
    inputs=("datasets/cube/0/train.bundle",),
    outputs=("trained_classifiers/cube-0.bundle",),
    dependencies=frozenset({40}),
)
# A dataset bundle of a stage that is off: no job of the plan makes it.
DATASET = job(
    "dataset_generation",
    40,
    JobIdentity(
        stage="dataset_generation",
        name="cube",
        dataset_key="cube",
        dataset_realization_index=0,
    ),
    status="reuse",
    reasons=(StatusReason(kind="outputs_up_to_date"),),
    outputs=("datasets/cube/0/train.bundle",),
)
PLOT = job(
    "plot_eval_perf",
    30,
    None,
    wildcards={"metric": "accuracy"},
    reasons=(StatusReason(kind="missing_output", files=("plots/acc.pdf",)),),
    inputs=("eval/alpha/0.csv", "eval/beta/0.csv"),
    outputs=("plots/acc.pdf",),
    dependencies=frozenset({10}),
)
PLAN = PipelinePlan(
    jobs=[
        training(1, "alpha", 0),
        training(2, "alpha", 1, status="reuse"),
        training(3, "beta", 0),
        EVALUATION,
        CLASSIFIER,
        DATASET,
        PLOT,
    ]
)


def outline(nodes: tuple[GroupNode | JobNode, ...]) -> list[object]:
    """Return labels and counts, nested as the tree is."""
    return [
        (node.label, node.counts_label, outline(node.children))
        if isinstance(node, GroupNode)
        else (node.status, node.label)
        for node in nodes
    ]


def test_jobs_are_grouped_by_stage_method_and_dataset_in_pipeline_order() -> (
    None
):
    assert outline(preview_tree(PLAN)) == [
        (
            "dataset generation",
            "1 Reuse",
            [
                (
                    "Shared prerequisites",
                    "1 Reuse",
                    [
                        (
                            "cube",
                            "1 Reuse",
                            [
                                (
                                    "reuse",
                                    "dataset_generation: realization 0",
                                )
                            ],
                        )
                    ],
                )
            ],
        ),
        (
            "classifier training",
            "1 Reuse",
            [
                (
                    "Shared prerequisites",
                    "1 Reuse",
                    [
                        (
                            "cube",
                            "1 Reuse",
                            [("reuse", "train_classifier: realization 0")],
                        )
                    ],
                )
            ],
        ),
        (
            "training",
            "2 Run, 1 Reuse",
            [
                (
                    "alpha",
                    "1 Run, 1 Reuse",
                    [
                        (
                            "cube",
                            "1 Run, 1 Reuse",
                            [
                                (
                                    "run",
                                    "train_method: realization 0, train seed 0, train hard budget 4",
                                ),
                                (
                                    "reuse",
                                    "train_method: realization 1, train seed 0, train hard budget 4",
                                ),
                            ],
                        )
                    ],
                ),
                (
                    "beta",
                    "1 Run",
                    [
                        (
                            "cube",
                            "1 Run",
                            [
                                (
                                    "run",
                                    "train_method: realization 0, train seed 0, train hard budget 4",
                                )
                            ],
                        )
                    ],
                ),
            ],
        ),
        (
            "evaluation",
            "1 Run",
            [
                (
                    "alpha",
                    "1 Run",
                    [
                        (
                            "cube",
                            "1 Run",
                            [
                                (
                                    "run",
                                    "eval_method: realization 0, train seed 0, eval seed 0, train hard budget 4, eval hard budget 2",
                                )
                            ],
                        )
                    ],
                )
            ],
        ),
        (
            "visualization",
            "1 Run",
            [
                (
                    "Across methods",
                    "1 Run",
                    [
                        (
                            "Across datasets",
                            "1 Run",
                            [("run", "plot_eval_perf: metric=accuracy")],
                        )
                    ],
                )
            ],
        ),
    ]


def test_each_job_counts_once_in_the_plan_summary() -> None:
    assert plan_summary(PLAN) == "7 jobs: 4 Run, 3 Reuse"
    assert sum(group.job_count for group in preview_tree(PLAN)) == 7


def test_a_group_summary_counts_its_jobs_by_status_and_subgroup() -> None:
    training_group = preview_tree(PLAN)[2]

    summary = group_summary(training_group)

    assert "Stage: training" in summary
    assert "3 jobs: 2 Run, 1 Reuse" in summary
    assert "By method: alpha (1 Run, 1 Reuse), beta (1 Run)" in summary


def test_job_details_explain_a_run_and_its_budgets() -> None:
    details = dict(job_details(EVALUATION, PLAN))

    assert details["Status"] == "Run"
    assert details["Reason"] == (
        "Inputs another job of this plan rewrites: alpha/0.bundle\n"
        "Params changed since the last run: eval_hard_budget"
    )
    assert details["Inputs"] == "alpha/0.bundle\ndatasets/cube/0/test.bundle"
    assert details["Outputs"] == "eval/alpha/0.csv"
    assert details["Training hard budget"] == "4"
    assert details["Evaluation hard budget"] == "2"
    assert details["Training soft-budget parameter"] == "not applicable"
    assert details["Evaluation soft-budget parameter"] == "not applicable"


def relations(
    job: PlannedJob, plan: PipelinePlan, view: PlanView = DEFAULT_VIEW
) -> list[tuple[str, JobStatus, str, bool]]:
    return [
        (
            relation.direction,
            relation.job.status,
            relation.title,
            relation.shown,
        )
        for relation in job_relations(job, plan, view)
    ]


def test_job_relations_name_the_jobs_it_needs_and_the_jobs_needing_it() -> (
    None
):
    assert relations(EVALUATION, PLAN) == [
        (
            "needs",
            "reuse",
            "train_classifier: realization 0 (masked_mlp_classifier, cube)",
            True,
        ),
        (
            "needs",
            "run",
            "train_method: realization 0, train seed 0, train hard budget 4 "
            "(alpha, cube)",
            True,
        ),
        ("needed_by", "run", "plot_eval_perf: metric=accuracy", True),
    ]
    # A job no job of the plan needs is needed by none.
    assert [direction for direction, _, _, _ in relations(PLOT, PLAN)] == [
        "needs"
    ]


def test_job_relations_tell_the_jobs_the_view_hides() -> None:
    view = PlanView(stage="evaluation")

    assert [
        (direction, shown)
        for direction, _, _, shown in relations(EVALUATION, PLAN, view)
    ] == [("needs", False), ("needs", False), ("needed_by", False)]
    assert relations(PLOT, PLAN, PlanView(search="eval_method"))[0][3]


def test_job_details_explain_a_reuse_and_its_consuming_methods() -> None:
    details = dict(job_details(CLASSIFIER, PLAN))

    assert details["Status"] == "Reuse"
    assert details["Reason"] == (
        "The outputs it is needed for exist and are up to date, and nothing "
        "it needs is rerun"
    )
    assert details["Method"] == "not applicable"
    # The job record's name is a classifier's, not a method's.
    assert details["Classifier"] == "masked_mlp_classifier"
    assert details["Consuming methods"] == "alpha, beta"


def test_a_per_method_classifier_belongs_to_its_method() -> None:
    alphas_classifier = job(
        "train_classifier_for_method",
        21,
        JobIdentity(
            stage="classifier_training",
            name="masked_mlp_classifier",
            dataset_key="cube",
            dataset_realization_index=0,
        ),
        wildcards={
            "method": "alpha",
            "dataset": "cube",
            "dataset_realization_index": "0",
        },
    )
    plan = PipelinePlan(jobs=[*PLAN.jobs, alphas_classifier])

    assert leaves(preview_tree(plan, PlanView(grouping=("method",))))[:2] == [
        ("alpha", "eval_method"),
        ("alpha", "train_classifier_for_method"),
    ]
    details = dict(job_details(alphas_classifier, plan))
    assert details["Method"] == "alpha"
    # The job record's name stays the classifier's.
    assert details["Classifier"] == "masked_mlp_classifier"
    assert "Consuming methods" not in details
    assert alphas_classifier in shown_jobs(plan, PlanView(method="alpha"))
    assert alphas_classifier not in shown_jobs(plan, PlanView(method="beta"))


def test_job_details_show_inputs_no_job_of_the_plan_makes() -> None:
    plan = PipelinePlan(jobs=[job for job in PLAN.jobs if job is not DATASET])

    details = dict(job_details(CLASSIFIER, plan))

    assert details["Existing inputs"] == "datasets/cube/0/train.bundle"
    # Its prerequisite made by no job is no relation; its consumers are.
    assert [
        relation.job
        for relation in job_relations(CLASSIFIER, plan, PlanView())
    ] == [
        EVALUATION,
        training(1, "alpha", 0),
        training(3, "beta", 0),
        training(2, "alpha", 1, status="reuse"),
    ]
    assert all(
        relation.direction == "needed_by"
        for relation in job_relations(CLASSIFIER, plan, PlanView())
    )
    assert dict(job_details(CLASSIFIER, PLAN))["Existing inputs"] == "none"


def test_job_details_say_which_dimensions_a_job_has_no_identity_for() -> None:
    details = dict(job_details(PLOT, PLAN))

    assert details["Job record"] == "none: the rule writes no job record"
    assert details["Method"] == "not applicable"
    assert details["Dataset"] == "not applicable"
    assert details["Wildcards"] == "metric=accuracy"
    assert "Classifier" not in details
    assert "Consuming methods" not in details
    assert "Training hard budget" not in details


def test_job_details_show_the_allocation_a_job_requests() -> None:
    details = dict(
        job_details(job("merge_eval_perf", 4, None, cpus=None), PLAN)
    )

    assert details["Device"] == "cpu"
    assert details["CPUs"] == "cluster default"
    assert details["GPUs"] == "0"
    assert details["Time limit"] == "30 minutes"
    assert details["Placement"] == "none requested"


def test_job_details_show_where_a_job_requests_to_be_placed() -> None:
    placed = job(
        "plot_time",
        5,
        None,
        placement={"slurm_partition": "gpu-queue", "gres": "gpu:T4:1"},
    )

    assert dict(job_details(placed, PLAN))["Placement"] == (
        "slurm_partition=gpu-queue\ngres=gpu:T4:1"
    )


def test_a_jobs_shell_command_is_shown_or_its_absence_explained() -> None:
    assert (
        shell_command_text(job("plot_time", 6, None, shell_command="plot x"))
        == "plot x"
    )
    absent = shell_command_text(job("plot_time", 6, None))
    assert absent.startswith("No shell command")
    assert "runs nothing" in absent


def leaves(nodes: tuple[GroupNode | JobNode, ...]) -> list[tuple[str, ...]]:
    """Return each job's group labels, top down, then its rule."""
    return [
        path
        for node in nodes
        for path in (
            [(node.label, *below) for below in leaves(node.children)]
            if isinstance(node, GroupNode)
            else [(node.job.rule,)]
        )
    ]


def test_jobs_can_be_grouped_by_any_ordered_subset_of_the_dimensions() -> None:
    by_dataset_then_method = preview_tree(
        PLAN, PlanView(grouping=("dataset", "method"))
    )

    assert leaves(by_dataset_then_method) == [
        ("cube", "alpha", "eval_method"),
        ("cube", "alpha", "train_method"),
        ("cube", "alpha", "train_method"),
        ("cube", "beta", "train_method"),
        ("cube", "Shared prerequisites", "dataset_generation"),
        ("cube", "Shared prerequisites", "train_classifier"),
        ("Across datasets", "Across methods", "plot_eval_perf"),
    ]
    assert leaves(preview_tree(PLAN, PlanView(grouping=("method",)))) == [
        ("alpha", "eval_method"),
        ("alpha", "train_method"),
        ("alpha", "train_method"),
        ("beta", "train_method"),
        ("Shared prerequisites", "dataset_generation"),
        ("Shared prerequisites", "train_classifier"),
        ("Across methods", "plot_eval_perf"),
    ]


def test_the_groupings_are_every_ordered_subset_default_first() -> None:
    assert len(GROUPINGS) == 15
    assert len(set(GROUPINGS)) == 15
    assert GROUPINGS[0] == ("stage", "method", "dataset")
    assert ("stage", "dataset", "method") in GROUPINGS
    assert ("dataset", "method") in GROUPINGS
    assert ("method",) in GROUPINGS


@pytest.mark.parametrize("grouping", GROUPINGS)
def test_every_grouping_shows_each_job_once(
    grouping: tuple[Dimension, ...],
) -> None:
    tree = preview_tree(PLAN, PlanView(grouping=grouping))

    assert sum(group.job_count for group in tree) == len(PLAN.jobs)
    assert len(leaves(tree)) == len(PLAN.jobs)
    assert all(len(path) == len(grouping) + 1 for path in leaves(tree))


def test_a_grouping_is_read_from_comma_separated_dimensions() -> None:
    assert parse_grouping("dataset,method") == ("dataset", "method")
    assert parse_grouping("stage") == ("stage",)


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("", "''"),
        ("stage,budget", "'budget'"),
        ("method,stage,method", "'method'"),
        ("stage,", "''"),
    ],
)
def test_a_grouping_names_known_dimensions_once(text: str, named: str) -> None:
    with pytest.raises(ValueError, match=named):
        parse_grouping(text)


# A pretrained model only alpha's third training needs: a shared
# prerequisite unrelated to beta.
PRETRAINING = job(
    "pretrain_model",
    50,
    JobIdentity(
        stage="pretraining",
        name="shared_encoder",
        dataset_key="cube",
        dataset_realization_index=2,
    ),
    status="reuse",
)
FILTERED_PLAN = PipelinePlan(
    jobs=[
        *PLAN.jobs,
        PRETRAINING,
        replace(training(5, "alpha", 2), dependencies=frozenset({50})),
    ]
)


def shown(view: PlanView) -> list[int]:
    return sorted(job.jobid for job in shown_jobs(FILTERED_PLAN, view))


def test_without_filters_or_search_every_job_is_shown() -> None:
    assert shown(PlanView()) == [1, 2, 3, 5, 10, 20, 30, 40, 50]


def test_stage_dataset_and_status_filters_are_strict() -> None:
    assert shown(PlanView(stage="training")) == [1, 2, 3, 5]
    assert shown(PlanView(dataset="cube")) == [1, 2, 3, 5, 10, 20, 40, 50]
    assert shown(PlanView(status="reuse")) == [2, 20, 40, 50]
    assert shown(PlanView(stage="training", status="reuse")) == [2]


def test_a_method_filter_keeps_the_methods_prerequisites_only() -> None:
    # Beta's training needs the classifier and the dataset, not alpha's
    # pretrained model; the plot needs beta's outputs but is no prerequisite.
    assert shown(PlanView(method="beta")) == [3, 20, 40]
    assert shown(PlanView(method="alpha")) == [1, 2, 5, 10, 20, 40, 50]


def test_other_filters_stay_strict_on_a_methods_prerequisites() -> None:
    assert shown(PlanView(method="alpha", stage="classifier training")) == [20]
    assert shown(PlanView(method="alpha", status="run")) == [1, 5, 10]
    assert shown(PlanView(method="beta", dataset="cube", status="reuse")) == [
        20,
        40,
    ]


def test_search_narrows_by_every_term_regardless_of_case() -> None:
    assert shown(PlanView(search="beta")) == [3]
    assert shown(PlanView(search="ALPHA eval")) == [10]
    assert shown(PlanView(search="classifier")) == [20]
    assert shown(PlanView(search="metric=accuracy")) == [30]
    assert shown(PlanView(search="pretrain", method="alpha")) == [50]
    assert shown(PlanView(search="pretrain", method="beta")) == []
    assert shown(PlanView(search="nothing like it")) == []


def test_the_tree_groups_the_shown_jobs_and_counts_them() -> None:
    tree = preview_tree(FILTERED_PLAN, PlanView(method="beta"))

    assert [(group.label, group.counts_label) for group in tree] == [
        ("dataset generation", "1 Reuse"),
        ("classifier training", "1 Reuse"),
        ("training", "1 Run"),
    ]


def test_the_summary_tells_the_shown_jobs_from_the_whole_plan() -> None:
    assert view_summary(FILTERED_PLAN, PlanView(grouping=("method",))) == (
        "9 jobs: 5 Run, 4 Reuse"
    )
    assert view_summary(FILTERED_PLAN, PlanView(method="beta")) == (
        "Shown: 3 of 9 jobs: 1 Run, 2 Reuse | Whole plan: 9 jobs: 5 Run, 4 Reuse"
    )
    assert view_summary(FILTERED_PLAN, PlanView(search="nothing")) == (
        "Shown: no job matches the filters and search | Whole plan: 9 jobs: "
        "5 Run, 4 Reuse"
    )


def test_the_filters_offer_the_values_and_statuses_of_the_plan() -> None:
    assert filter_values(FILTERED_PLAN, "stage") == [
        "dataset generation",
        "classifier training",
        "pretraining",
        "training",
        "evaluation",
        "visualization",
    ]
    assert filter_values(FILTERED_PLAN, "method") == ["alpha", "beta"]
    assert filter_values(FILTERED_PLAN, "dataset") == ["cube"]
    assert plan_statuses(FILTERED_PLAN) == ["run", "reuse"]
    assert plan_statuses(PipelinePlan(jobs=[training(1, "alpha", 0)])) == [
        "run"
    ]


# A release preview: the classifier and alpha's training on realization 0
# wait for a regenerated dataset's check; beta's training runs.
REGENERATED = job(
    "dataset_generation",
    50,
    JobIdentity(
        stage="dataset_generation",
        name="cube",
        dataset_key="cube",
        dataset_realization_index=0,
    ),
    reasons=(
        StatusReason(
            kind="missing_output", files=("datasets/cube/0/train.bundle",)
        ),
    ),
    outputs=("datasets/cube/0/train.bundle",),
)
CONDITIONAL_CLASSIFIER = job(
    "train_classifier",
    51,
    CLASSIFIER.identity,
    status="conditional_reuse",
    reasons=(
        StatusReason(
            kind="pending_dataset_check",
            files=("datasets/cube/0/train.bundle",),
        ),
    ),
    inputs=("datasets/cube/0/train.bundle",),
    outputs=("trained_classifiers/cube-0.bundle",),
    dependencies=frozenset({50}),
)
CONDITIONAL_TRAINING = job(
    "train_method",
    52,
    JobIdentity(
        stage="training",
        name="alpha",
        dataset_key="cube",
        dataset_realization_index=0,
        train_hard_budget=4,
    ),
    status="conditional_reuse",
    reasons=(
        StatusReason(
            kind="pending_dataset_check",
            files=("datasets/cube/0/train.bundle",),
        ),
        StatusReason(
            kind="conditional_input",
            files=("trained_classifiers/cube-0.bundle",),
        ),
    ),
    dependencies=frozenset({50, 51}),
)
RELEASE_PLAN = PipelinePlan(
    jobs=[
        REGENERATED,
        CONDITIONAL_CLASSIFIER,
        CONDITIONAL_TRAINING,
        training(53, "beta", 0),
    ]
)


def test_conditional_reuse_is_counted_apart_from_run_and_reuse() -> None:
    assert plan_summary(RELEASE_PLAN) == (
        "4 jobs: 2 Run, 0 Reuse, 2 Conditional reuse"
    )
    # A plan without it does not mention it.
    assert plan_summary(PLAN) == "7 jobs: 4 Run, 3 Reuse"
    assert [
        (group.label, group.counts_label)
        for group in preview_tree(RELEASE_PLAN)
    ] == [
        ("dataset generation", "1 Run"),
        ("classifier training", "1 Conditional reuse"),
        ("training", "1 Run, 1 Conditional reuse"),
    ]


def test_job_details_name_the_dataset_checks_a_conditional_reuse_waits_for() -> (
    None
):
    details = dict(job_details(CONDITIONAL_TRAINING, RELEASE_PLAN))

    assert details["Status"] == "Conditional reuse"
    assert details["Reason"] == (
        "Released outputs, reused during execution only if these regenerated "
        "dataset bundles match the release's recorded content: "
        "datasets/cube/0/train.bundle\n"
        "Reads outputs that are themselves conditionally reused: "
        "trained_classifiers/cube-0.bundle"
    )
    assert [
        (relation.job.status, relation.job.rule)
        for relation in job_relations(
            CONDITIONAL_TRAINING, RELEASE_PLAN, PlanView()
        )
    ] == [
        ("run", "dataset_generation"),
        ("conditional_reuse", "train_classifier"),
    ]


def test_a_plan_with_conditional_reuse_warns_that_its_checks_are_pending() -> (
    None
):
    warning = release_warning(RELEASE_PLAN)

    assert warning is not None
    assert "regenerated datasets must match" in warning
    assert "not checked" in warning
    assert release_warning(PLAN) is None


def test_conditional_reuse_can_be_filtered_like_any_status() -> None:
    view = PlanView(status="conditional_reuse")

    assert plan_statuses(RELEASE_PLAN) == ["run", "conditional_reuse"]
    assert sorted(job.jobid for job in shown_jobs(RELEASE_PLAN, view)) == [
        51,
        52,
    ]
    assert view_summary(RELEASE_PLAN, view) == (
        "Shown: 2 of 4 jobs: 2 Conditional reuse | Whole plan: 4 jobs: "
        "2 Run, 0 Reuse, 2 Conditional reuse"
    )
    # Alpha's conditionally reused training keeps what it needs.
    assert sorted(
        job.jobid for job in shown_jobs(RELEASE_PLAN, PlanView(method="alpha"))
    ) == [50, 51, 52]


# A reused training whose identity, allocation and shell command could not
# be evaluated
UNAVAILABLE = replace(
    job(
        "train_method",
        60,
        None,
        status="reuse",
        reasons=(StatusReason(kind="outputs_up_to_date"),),
        wildcards={"method": "gamma"},
        inputs=("datasets/cube/0/train.bundle",),
        outputs=("gamma/0.bundle",),
        cpus=None,
    ),
    device=None,
    gpus=None,
    time_limit_minutes=None,
    unavailable="ValueError: Cannot size the inputs of gamma",
)
UNAVAILABLE_PLAN = PipelinePlan(jobs=[*PLAN.jobs, UNAVAILABLE])


def test_a_job_with_an_unavailable_identity_is_grouped_apart() -> None:
    tree = preview_tree(UNAVAILABLE_PLAN)

    assert outline(tree[-1:]) == [
        (
            "Identity unavailable",
            "1 Reuse",
            [
                (
                    "Identity unavailable",
                    "1 Reuse",
                    [
                        (
                            "Identity unavailable",
                            "1 Reuse",
                            [("reuse", "train_method: method=gamma")],
                        )
                    ],
                )
            ],
        )
    ]
    assert plan_summary(UNAVAILABLE_PLAN) == "8 jobs: 4 Run, 4 Reuse"
    assert leaves(
        preview_tree(UNAVAILABLE_PLAN, PlanView(grouping=("method",)))
    )[-2:] == [
        ("Across methods", "plot_eval_perf"),
        ("Identity unavailable", "train_method"),
    ]


def test_job_details_say_why_a_jobs_identity_and_allocation_are_unavailable() -> (
    None
):
    details = dict(job_details(UNAVAILABLE, UNAVAILABLE_PLAN))

    assert details["Status"] == "Reuse"
    assert details["Rule"] == "train_method"
    assert details["Wildcards"] == "method=gamma"
    assert details["Unavailable"] == (
        "Evaluating its params or resources failed, so its identity, "
        "allocation and shell command are unknown:\n"
        "ValueError: Cannot size the inputs of gamma"
    )
    for label in [
        "Stage",
        "Method",
        "Dataset",
        "Device",
        "CPUs",
        "GPUs",
        "Time limit",
        "Placement",
    ]:
        assert details[label] == "unavailable"
    assert "Job record" not in details
    assert details["Inputs"] == "datasets/cube/0/train.bundle"
    assert details["Outputs"] == "gamma/0.bundle"
    assert shell_command_text(UNAVAILABLE) == (
        "Shell command unavailable: ValueError: Cannot size the inputs of gamma"
    )


def test_a_rule_without_job_records_and_a_stage_is_refused() -> None:
    plan = PipelinePlan(jobs=[*PLAN.jobs, job("plot_novelty", 70, None)])

    with pytest.raises(ValueError, match="'plot_novelty'"):
        preview_tree(plan)
