"""
Planners standing in for `python -m afabench.pipeline_plan` (#158).

The preview plans in a child process it starts with an interpreter. These
fixtures give it a Python script as the interpreter, so a plan arrives,
planning fails or hangs quickly, without Snakemake.
"""

import sys
from pathlib import Path

import pytest

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.plan_json import dump_plan
from afabench.pipeline_plan.planning import PipelinePlan, PlannedJob

# The script, run as `<interpreter> -m afabench.pipeline_plan --output
# <path> -- <arguments>`, reads the output path from its arguments.
OUTPUT_ARGUMENT = "sys.argv[sys.argv.index('--output') + 1]"


def scheduled(
    rule: str,
    jobid: int,
    identity: JobIdentity | None,
    dependencies: frozenset[int] | None = None,
) -> PlannedJob:
    return PlannedJob(
        rule=rule,
        wildcards={},
        identity=identity,
        device="cpu",
        cpus=2,
        gpus=0,
        jobid=jobid,
        dependencies=dependencies or frozenset(),
    )


def training(jobid: int, method: str, realization: int) -> PlannedJob:
    return scheduled(
        "train_method",
        jobid,
        JobIdentity(
            stage="training",
            name=method,
            dataset_key="cube",
            dataset_realization_index=realization,
            train_hard_budget=4,
        ),
    )


@pytest.fixture
def plan() -> PipelinePlan:
    """Two training jobs of alpha, one of beta, and alpha's evaluation."""
    return PipelinePlan(
        scheduled_jobs=[
            training(1, "alpha", 0),
            training(2, "alpha", 1),
            training(3, "beta", 0),
            scheduled(
                "eval_method",
                4,
                JobIdentity(
                    stage="evaluation",
                    name="alpha",
                    dataset_key="cube",
                    dataset_realization_index=0,
                    train_hard_budget=4,
                    eval_hard_budget=2,
                ),
                dependencies=frozenset({1}),
            ),
        ]
    )


def interpreter(tmp_path: Path, name: str, body: str) -> list[str]:
    script = tmp_path / f"{name}.py"
    script.write_text("import sys, time, shutil\n" + body)
    return [sys.executable, str(script)]


@pytest.fixture
def planning_interpreter(tmp_path: Path, plan: PipelinePlan) -> list[str]:
    """Return an interpreter whose planning child writes `plan`."""
    encoded = tmp_path / "encoded_plan.json"
    encoded.write_text(dump_plan(plan))
    return interpreter(
        tmp_path,
        "planning",
        f"print('Building DAG of jobs...')\n"
        f"shutil.copyfile({str(encoded)!r}, {OUTPUT_ARGUMENT})\n",
    )


@pytest.fixture
def failing_interpreter(tmp_path: Path) -> list[str]:
    """Return an interpreter whose planning child fails as Snakemake does."""
    return interpreter(
        tmp_path,
        "failing",
        "print('Building DAG of jobs...')\n"
        "print('MissingInputException: cube/0/train.bundle', file=sys.stderr)\n"
        "sys.exit(1)\n",
    )


@pytest.fixture
def hanging_interpreter(tmp_path: Path) -> tuple[list[str], Path]:
    """Return an interpreter whose planning child never ends, and its pid file."""
    pid_file = tmp_path / "planner.pid"
    return interpreter(
        tmp_path,
        "hanging",
        "import os\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n",
    ), pid_file
