"""
The shared planner turns Snakemake arguments into a pipeline plan (#157).

A minimal Snakefile stands in for the pipeline: an actionless target rule
over a job that writes a job record and the job preparing its input. The
pipeline itself is covered by `test/workflow/test_compute_estimate_planning.py`.
"""

from pathlib import Path

import pytest

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.planning import InvocationError, plan_invocation

SNAKEFILE = """
rule all:
    input: "alpha.txt", "beta.txt"

rule train_method:
    input: "{method}.data"
    output: "{method}.txt"
    params:
        job_record=lambda wc: (
            "python -m afabench.core.job_record --no-smoke-test "
            "--record r --failed-record f --stage training --device cpu "
            f"--name {wc.method} --dataset-key cube --"
        ),
    shell: "touch {output}"

rule prepare:
    output: "{method}.data"
    shell: "touch {output}"
"""


@pytest.fixture
def workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the workflow with beta's input already prepared."""
    (tmp_path / "Snakefile").write_text(SNAKEFILE)
    (tmp_path / "beta.data").write_text("prepared")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_the_plan_schedules_the_jobs_to_run_with_their_dependencies(
    workflow: Path,
) -> None:
    plan = plan_invocation(["--snakefile", "Snakefile", "--cores", "1"])

    jobs = {job.jobid: job for job in plan.scheduled_jobs}
    needed = {
        (job.rule, job.wildcards["method"]): sorted(
            (jobs[jobid].rule, jobs[jobid].wildcards["method"])
            for jobid in job.dependencies
        )
        for job in jobs.values()
    }
    # The target rule has no action and beta's input exists.
    assert needed == {
        ("prepare", "alpha"): [],
        ("train_method", "alpha"): [("prepare", "alpha")],
        ("train_method", "beta"): [],
    }
    identities = {
        (job.rule, job.wildcards["method"]): job.identity
        for job in plan.scheduled_jobs
    }
    assert identities[("train_method", "alpha")] == JobIdentity(
        stage="training", name="alpha", dataset_key="cube"
    )
    assert identities[("prepare", "alpha")] is None
    # Planning runs nothing.
    assert not [
        name
        for name in ["alpha.data", "alpha.txt", "beta.txt"]
        if (workflow / name).exists()
    ]


@pytest.mark.usefixtures("workflow")
def test_an_invocation_snakemake_rejects_fails_planning() -> None:
    with pytest.raises(InvocationError):
        plan_invocation(["--snakefile", "Snakefile", "--set-resources", "x"])
