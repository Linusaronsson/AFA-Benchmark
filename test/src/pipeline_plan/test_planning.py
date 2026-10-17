"""
The shared planner turns Snakemake arguments into a pipeline plan (#157).

A minimal Snakefile stands in for the pipeline: an actionless target rule
over a job that writes a job record and the job preparing its input. The
pipeline itself is covered by `test/workflow/test_pipeline_plan.py`.
"""

from pathlib import Path

import pytest

from afabench.core.job_record import JobIdentity
from afabench.pipeline_plan.planning import (
    InvocationError,
    StatusReason,
    plan_invocation,
)

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


def test_the_plan_holds_the_jobs_to_run_and_the_jobs_reused(
    workflow: Path,
) -> None:
    plan = plan_invocation(["--snakefile", "Snakefile", "--cores", "1"])

    jobs = {(job.rule, job.wildcards["method"]): job for job in plan.jobs}
    by_id = {job.jobid: job for job in plan.jobs}
    needed = {
        key: sorted(
            (by_id[jobid].rule, by_id[jobid].wildcards["method"])
            for jobid in job.dependencies
        )
        for key, job in jobs.items()
    }
    # The target rule has no action and beta's input exists, so preparing
    # it is reused, not run.
    assert needed == {
        ("prepare", "alpha"): [],
        ("prepare", "beta"): [],
        ("train_method", "alpha"): [("prepare", "alpha")],
        ("train_method", "beta"): [("prepare", "beta")],
    }
    assert {key: job.status for key, job in jobs.items()} == {
        ("prepare", "alpha"): "run",
        ("prepare", "beta"): "reuse",
        ("train_method", "alpha"): "run",
        ("train_method", "beta"): "run",
    }
    assert jobs[("train_method", "alpha")].identity == JobIdentity(
        stage="training", name="alpha", dataset_key="cube"
    )
    assert jobs[("prepare", "alpha")].identity is None
    assert jobs[("train_method", "beta")].inputs == ("beta.data",)
    assert jobs[("train_method", "beta")].outputs == ("beta.txt",)
    # Planning runs nothing.
    assert not [
        name
        for name in ["alpha.data", "alpha.txt", "beta.txt"]
        if (workflow / name).exists()
    ]


@pytest.mark.usefixtures("workflow")
def test_the_plan_gives_snakemakes_reasons_for_each_status() -> None:
    plan = plan_invocation(["--snakefile", "Snakefile", "--cores", "1"])

    reasons = {
        (job.rule, job.wildcards["method"]): job.reasons for job in plan.jobs
    }
    assert reasons[("train_method", "beta")] == (
        StatusReason(kind="missing_output", files=("beta.txt",)),
    )
    assert reasons[("prepare", "alpha")] == (
        StatusReason(kind="missing_output", files=("alpha.data",)),
    )
    assert reasons[("prepare", "beta")] == (
        StatusReason(kind="outputs_up_to_date"),
    )


@pytest.mark.usefixtures("workflow")
def test_the_scheduled_jobs_depend_on_scheduled_jobs_only() -> None:
    plan = plan_invocation(["--snakefile", "Snakefile", "--cores", "1"])

    scheduled = {
        (job.rule, job.wildcards["method"]): job for job in plan.scheduled_jobs
    }
    assert set(scheduled) == {
        ("prepare", "alpha"),
        ("train_method", "alpha"),
        ("train_method", "beta"),
    }
    # The compute estimate chains scheduled jobs; a reused job is no link.
    assert scheduled[("train_method", "beta")].dependencies == frozenset()
    assert scheduled[("train_method", "alpha")].dependencies == {
        scheduled[("prepare", "alpha")].jobid
    }


@pytest.mark.usefixtures("workflow")
def test_an_invocation_snakemake_rejects_fails_planning() -> None:
    with pytest.raises(InvocationError):
        plan_invocation(["--snakefile", "Snakefile", "--set-resources", "x"])
