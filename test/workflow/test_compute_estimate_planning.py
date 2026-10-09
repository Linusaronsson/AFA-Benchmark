"""
`estimate-compute` plans an invocation's jobs with their allocations (#79).

The planner takes the same Snakemake arguments as a real run. These tests
compare the jobs of its per-job CSV with what the same arguments make
Snakemake submit to the fake `sbatch`, or with how a real invocation fails.
"""

import json
import re
from pathlib import Path

import pytest

from test.workflow.submission_harness import WorkflowHarness
from test.workflow.test_reference_results import (
    ALPHA_HARD_BUDGET_TABLE,
    adopter_workflow,
)

type Job = tuple[str, tuple[str, ...], str, int, int]


def submitted_allocation(args: list[str]) -> Job:
    """Return a captured submission's identity and allocation."""
    comment = args[args.index("--comment") + 1].removeprefix("rule_")
    rule, _, wildcards = comment.partition("_wildcards_")
    gpus = sum(
        int(request.rsplit(":", 1)[1])
        for request in args
        if re.match(r"--(gres|gpus)=", request)
    )
    cpus = next(
        int(arg.removeprefix("--cpus-per-task="))
        for arg in args
        if arg.startswith("--cpus-per-task=")
    )
    device = "cuda" if gpus else "cpu"
    return rule, tuple(sorted(wildcards.split("_"))), device, cpus, gpus


def wildcards(row: dict[str, str]) -> dict[str, str]:
    """Return a planned job's wildcards from its row of the per-job CSV."""
    return json.loads(row["wildcards"])


def planned_allocation(row: dict[str, str]) -> Job:
    """Return a planned job's identity, in the form of a submission comment."""
    # The SLURM plugin joins wildcard values with "_" and replaces "/".
    values = "_".join(
        value for value in wildcards(row).values() if value
    ).replace("/", "_")
    return (
        row["rule"],
        tuple(sorted(values.split("_"))),
        row["device"],
        int(row["cpus"]),
        int(row["gpus"]),
    )


def test_planned_jobs_match_the_submitted_jobs_and_allocations(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution"] = {"methods": {"alpha": {"training": "cuda"}}}
    profile = str(tmp_path / "workflow/profiles/site/examples/mixed-gres")

    plan = workflow.plan(
        "--workflow-profile", profile, target="all_train_methods"
    )
    workflow.submit_first_wave(
        2, "--workflow-profile", profile, target="all_train_methods"
    )

    assert plan.returncode == 0, plan.stdout + plan.stderr
    submitted = sorted(map(submitted_allocation, workflow.submissions()))
    assert (
        sorted(map(planned_allocation, workflow.planned_jobs())) == submitted
    )
    assert {job[2:] for job in submitted} == {("cuda", 8, 1), ("cpu", 8, 0)}


def test_only_the_remaining_jobs_are_planned_once_outputs_exist(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    trained = workflow.run("--executor", "local", target="all_train_methods")
    assert trained.returncode == 0, trained.stdout + trained.stderr

    plan = workflow.plan("--executor", "local")

    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert sorted(
        (job["rule"], wildcards(job)["method"])
        for job in workflow.planned_jobs()
    ) == [("eval_method", "alpha"), ("eval_method", "beta")]


def test_reference_methods_are_never_planned(tmp_path: Path) -> None:
    workflow = adopter_workflow(tmp_path, ALPHA_HARD_BUDGET_TABLE)

    plan = workflow.plan("--executor", "local", target="all")

    assert plan.returncode == 0, plan.stdout + plan.stderr
    jobs = workflow.planned_jobs()
    assert {
        wildcards(job)["method"] for job in jobs if "method" in wildcards(job)
    } == {"beta"}
    assert "merge_eval_perf" in {job["rule"] for job in jobs}


def test_set_resources_overrides_are_reflected_in_the_planned_allocation(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["execution"] = {"methods": {"alpha": {"training": "cuda"}}}

    plan = workflow.plan(
        "--workflow-profile",
        str(tmp_path / "workflow/profiles/site/examples/mixed-gres"),
        "--set-resources",
        "train_method:cpus_per_task=3",
        target="all_train_methods",
    )

    assert plan.returncode == 0, plan.stdout + plan.stderr
    assert sorted(
        (wildcards(job)["method"], job["device"], job["cpus"], job["gpus"])
        for job in workflow.planned_jobs()
    ) == [("alpha", "cuda", "3", "1"), ("beta", "cpu", "3", "0")]


@pytest.mark.parametrize(
    ("change", "options", "diagnostic"),
    [
        ({}, ["--executor", "slurm"], "needs an execution_site allocation"),
        (
            {},
            ["--dry-run", "--set-resources", "train_method:gpu=1"],
            "Conflicting allocation",
        ),
        (
            {"execution": {"defaults": {"training": "tpu"}}},
            ["--dry-run"],
            "Invalid execution choice 'tpu'",
        ),
        ({}, ["--set-resources", "train_method"], "Invalid resource"),
    ],
)
def test_invalid_invocation_fails_the_same_way_as_a_real_one(
    tmp_path: Path,
    change: dict[str, object],
    options: list[str],
    diagnostic: str,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config.update(change)

    plan = workflow.plan(*options)
    run = workflow.run(*options)

    assert run.returncode != 0
    assert diagnostic in run.stdout + run.stderr
    assert plan.returncode == run.returncode
    assert diagnostic in plan.stdout + plan.stderr
    assert not workflow.planned_jobs_path.exists()
    assert workflow.submissions() == []
