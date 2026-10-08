"""
`estimate-compute` estimates a harness run from its own job records (#83).

The estimate takes the same arguments as the run, so every job it plans
has a job record with the same identity and allocation.
"""

from pathlib import Path

import pytest

from afabench.core.job_duration_table import load_job_duration_table
from test.workflow.submission_harness import WorkflowHarness

EVALUATION_CPUS = 4


def test_an_estimate_of_a_recorded_run_matches_every_job_exactly(
    tmp_path: Path,
) -> None:
    workflow = WorkflowHarness(tmp_path)
    workflow.config["smoke_test"] = False
    workflow.config["execution"] = {"methods": {"alpha": {"training": "cuda"}}}
    options = (
        "--executor",
        "local",
        "--workflow-profile",
        str(tmp_path / "extra/workflow/profiles/mixed-gres"),
        # Replaces the profile's set-resources: training gets its default
        # of 1 CPU.
        "--set-resources",
        f"eval_method:cpus_per_task={EVALUATION_CPUS}",
    )
    run = workflow.run(*options)
    assert run.returncode == 0, run.stdout + run.stderr

    # The run's jobs again, now that their job records exist
    plan = workflow.plan(*options, "--forcerun", "train_method")

    assert plan.returncode == 0, plan.stdout + plan.stderr
    jobs = workflow.planned_jobs()
    assert sorted(
        (job["stage"], job["name"], job["device"], job["cpus"], job["gpus"])
        for job in jobs
    ) == [
        ("evaluation", "alpha", "cpu", "4", "0"),
        ("evaluation", "beta", "cpu", "4", "0"),
        ("training", "alpha", "cuda", "1", "1"),
        ("training", "beta", "cpu", "1", "0"),
    ]
    assert {job["match_level"] for job in jobs} == {"exact"}
    records = load_job_duration_table(tmp_path / "extra/output")
    seconds = {
        (stage, name): duration
        for stage, name, duration in zip(
            records["stage"],
            records["name"],
            records["job_duration_seconds"],
            strict=True,
        )
    }
    for job in jobs:
        duration = seconds[job["stage"], job["name"]]
        assert float(job["mean_job_duration_seconds"]) == pytest.approx(
            duration
        )
        assert float(job["mean_core_hours"]) == pytest.approx(
            duration / 3600 * int(job["cpus"])
        )
        assert float(job["mean_gpu_hours"]) == pytest.approx(
            duration / 3600 * int(job["gpus"])
        )
    assert "4 planned jobs: 4 exact, 0 pooled, 0 unestimated" in plan.stdout
