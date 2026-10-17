"""The preview's presentation of a pipeline plan, apart from its widgets (#158)."""

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.planning import (
    JobStatus,
    PipelinePlan,
    PlannedJob,
    StatusReason,
)
from afabench.pipeline_preview.model import (
    GroupNode,
    JobNode,
    group_summary,
    job_details,
    plan_summary,
    preview_tree,
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


def test_job_details_explain_a_run_and_name_what_it_needs() -> None:
    details = dict(job_details(EVALUATION, PLAN))

    assert details["Status"] == "Run"
    assert details["Reason"] == (
        "Inputs another job of this plan rewrites: alpha/0.bundle\n"
        "Params changed since the last run: eval_hard_budget"
    )
    assert details["Inputs"] == "alpha/0.bundle\ndatasets/cube/0/test.bundle"
    assert details["Outputs"] == "eval/alpha/0.csv"
    assert details["Needs"] == (
        "Reuse train_classifier: realization 0\n"
        "Run train_method: realization 0, train seed 0, train hard budget 4"
    )
    assert details["Needed by"] == "Run plot_eval_perf: metric=accuracy"
    assert details["Training hard budget"] == "4"
    assert details["Evaluation hard budget"] == "2"


def test_job_details_explain_a_reuse_and_its_consuming_methods() -> None:
    details = dict(job_details(CLASSIFIER, PLAN))

    assert details["Status"] == "Reuse"
    assert details["Reason"] == (
        "The outputs it is needed for exist and are up to date, and nothing "
        "it needs is rerun"
    )
    assert details["Method"] == "not applicable"
    assert details["Consuming methods"] == "alpha, beta"


def test_job_details_show_inputs_no_job_of_the_plan_makes() -> None:
    plan = PipelinePlan(jobs=[job for job in PLAN.jobs if job is not DATASET])

    details = dict(job_details(CLASSIFIER, plan))

    assert details["Existing inputs"] == "datasets/cube/0/train.bundle"
    assert details["Needs"] == "no job of this plan"
    assert dict(job_details(CLASSIFIER, PLAN))["Existing inputs"] == "none"


def test_job_details_say_which_dimensions_a_job_has_no_identity_for() -> None:
    details = dict(job_details(PLOT, PLAN))

    assert details["Job record"] == "none: the rule writes no job record"
    assert details["Method"] == "not applicable"
    assert details["Dataset"] == "not applicable"
    assert details["Wildcards"] == "metric=accuracy"
    assert details["Needed by"] == "no job of this plan"
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
