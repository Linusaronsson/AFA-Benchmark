"""
`estimate-compute` plans an invocation's jobs with their allocations (#79).

The planner takes the same Snakemake arguments as a real run. These tests
compare its per-job CSV with what the same arguments make Snakemake submit
to the fake `sbatch`, or with how a real invocation fails.
"""

import re
from pathlib import Path

from test.workflow.submission_harness import WorkflowHarness

type Job = tuple[str, tuple[str, ...], str, int, int]


def submitted_job(args: list[str]) -> Job:
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


def planned_job(row: dict[str, str]) -> Job:
    """Return a planned job's identity, in the form of a submission comment."""
    allocation = {"rule", "device", "cpus", "gpus"}
    # The SLURM plugin joins wildcard values with "_" and replaces "/".
    values = "_".join(
        value for key, value in row.items() if key not in allocation and value
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
    profile = str(tmp_path / "extra/workflow/profiles/mixed-gres")

    plan = workflow.plan(
        "--workflow-profile", profile, target="all_train_methods"
    )
    workflow.submit_first_wave(
        2, "--workflow-profile", profile, target="all_train_methods"
    )

    assert plan.returncode == 0, plan.stdout + plan.stderr
    submitted = sorted(map(submitted_job, workflow.submissions()))
    assert sorted(map(planned_job, workflow.planned_jobs())) == submitted
    assert {job[2:] for job in submitted} == {("cuda", 8, 1), ("cpu", 8, 0)}
