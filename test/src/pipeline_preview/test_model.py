"""The preview's presentation of a pipeline plan, apart from its widgets (#158)."""

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.planning import PipelinePlan, PlannedJob
from afabench.pipeline_preview.model import (
    GroupNode,
    JobNode,
    group_summary,
    job_details,
    preview_tree,
)


def job(
    rule: str,
    jobid: int,
    identity: JobIdentity | None,
    wildcards: dict[str, str] | None = None,
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
        dependencies=dependencies or frozenset(),
        time_limit_minutes=30,
    )


def training(
    jobid: int, method: str, realization: int, hard_budget: int = 4
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
    dependencies=frozenset({1}),
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
)
PLOT = job("plot_eval_perf", 30, None, wildcards={"metric": "accuracy"})
PLAN = PipelinePlan(
    scheduled_jobs=[
        training(1, "alpha", 0),
        training(2, "alpha", 1),
        training(3, "beta", 0),
        EVALUATION,
        CLASSIFIER,
        PLOT,
    ]
)


def outline(nodes: tuple[GroupNode | JobNode, ...]) -> list[object]:
    """Return labels and counts, nested as the tree is."""
    return [
        (node.label, node.job_count, outline(node.children))
        if isinstance(node, GroupNode)
        else node.label
        for node in nodes
    ]


def test_jobs_are_grouped_by_stage_method_and_dataset_in_pipeline_order() -> (
    None
):
    assert outline(preview_tree(PLAN)) == [
        (
            "classifier training",
            1,
            [
                (
                    "no method",
                    1,
                    [("cube", 1, ["train_classifier: realization 0"])],
                )
            ],
        ),
        (
            "training",
            3,
            [
                (
                    "alpha",
                    2,
                    [
                        (
                            "cube",
                            2,
                            [
                                "train_method: realization 0, train seed 0, train hard budget 4",
                                "train_method: realization 1, train seed 0, train hard budget 4",
                            ],
                        )
                    ],
                ),
                (
                    "beta",
                    1,
                    [
                        (
                            "cube",
                            1,
                            [
                                "train_method: realization 0, train seed 0, train hard budget 4"
                            ],
                        )
                    ],
                ),
            ],
        ),
        (
            "evaluation",
            1,
            [
                (
                    "alpha",
                    1,
                    [
                        (
                            "cube",
                            1,
                            [
                                "eval_method: realization 0, train seed 0, eval seed 0, train hard budget 4, eval hard budget 2"
                            ],
                        )
                    ],
                )
            ],
        ),
        (
            "visualization",
            1,
            [
                (
                    "no method",
                    1,
                    [("no dataset", 1, ["plot_eval_perf: metric=accuracy"])],
                )
            ],
        ),
    ]


def test_a_group_summary_counts_its_jobs_and_names_its_subgroups() -> None:
    training_group = preview_tree(PLAN)[1]

    summary = group_summary(training_group)

    assert "Stage: training" in summary
    assert "3 scheduled jobs" in summary
    assert "alpha (2), beta (1)" in summary


def test_job_details_tell_training_from_evaluation_budgets() -> None:
    details = dict(job_details(EVALUATION, PLAN))

    assert details["Rule"] == "eval_method"
    assert details["Stage"] == "evaluation"
    assert details["Method"] == "alpha"
    assert details["Dataset"] == "cube"
    assert details["Dataset realization"] == "0"
    assert details["Training hard budget"] == "4"
    assert details["Evaluation hard budget"] == "2"
    assert details["Training soft-budget parameter"] == "not applicable"
    assert details["Evaluation soft-budget parameter"] == "not applicable"
    assert details["Needs"] == (
        "train_method: realization 0, train seed 0, train hard budget 4"
    )


def test_job_details_say_which_dimensions_a_job_has_no_identity_for() -> None:
    details = dict(job_details(PLOT, PLAN))

    assert details["Job record"] == "none: the rule writes no job record"
    assert details["Method"] == "not applicable"
    assert details["Dataset"] == "not applicable"
    assert details["Wildcards"] == "metric=accuracy"
    assert details["Needs"] == "nothing scheduled"
    assert "Training hard budget" not in details


def test_job_details_show_the_allocation_a_job_requests() -> None:
    details = dict(
        job_details(job("merge_eval_perf", 4, None, cpus=None), PLAN)
    )

    assert details["Device"] == "cpu"
    assert details["CPUs"] == "cluster default"
    assert details["GPUs"] == "0"
    assert details["Time limit"] == "30 minutes"
