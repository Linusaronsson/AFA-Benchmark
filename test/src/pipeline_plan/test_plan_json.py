"""A pipeline plan survives its JSON encoding, which crosses processes (#158)."""

import pytest

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.plan_json import (
    PlanJsonError,
    dump_plan,
    load_plan,
)
from afabench.pipeline_plan.planning import PipelinePlan, PlannedJob


def test_a_plan_is_read_back_unchanged_from_its_json() -> None:
    plan = PipelinePlan(
        scheduled_jobs=[
            PlannedJob(
                rule="plot_eval_perf",
                wildcards={},
                identity=None,
                device="cpu",
                cpus=None,
                gpus=0,
                jobid=3,
                dependencies=frozenset({7}),
            ),
            PlannedJob(
                rule="train_method",
                wildcards={
                    "method": "alpha",
                    "train_soft_budget_param": "0.5",
                },
                identity=JobIdentity(
                    stage="training",
                    name="alpha",
                    dataset_key="cube",
                    dataset_realization_index=0,
                    train_seed=1,
                    train_soft_budget_param=0.5,
                ),
                device="cuda",
                cpus=8,
                gpus=1,
                jobid=7,
                time_limit_minutes=90,
            ),
        ]
    )

    assert load_plan(dump_plan(plan)) == plan


def test_json_that_is_no_plan_is_refused() -> None:
    with pytest.raises(PlanJsonError, match="scheduled_jobs"):
        load_plan('{"jobs": []}')
