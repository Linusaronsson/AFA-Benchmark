"""
`estimate-compute` estimates a Snakemake invocation from job durations.

A minimal Snakefile stands in for the pipeline: its rule renders a job
record wrapper command, as the pipeline's computational rules do. The
pipeline itself is covered in
`test/workflow/test_compute_estimate_recorded_run.py`.
"""

import csv
import shutil
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from afabench.core.job_duration_table import write_job_duration_table
from afabench.core.job_record import JobIdentity
from scripts.compute_estimate.estimate_compute import app
from test import job_record_examples
from test.scripts.release_artifacts import (
    ALPHA,
    Catalog,
    write_catalog,
    write_job_record,
)
from test.scripts.test_release_manifest import (
    ALPHA_METHOD_RECORD,
    restore,
    save,
)

SNAKEFILE = """
rule all:
    input: "alpha.txt", "beta.txt"

rule train_method:
    output: "{method}.txt"
    threads: 2
    params:
        job_record=lambda wc: (
            "python -m afabench.core.job_record --no-smoke-test "
            "--record r --failed-record f --stage training --device cpu "
            f"--name {wc.method} --dataset-key cube --"
        ),
    shell: "touch {output}"
"""


@pytest.fixture
def workflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a workflow whose alpha job has a job record and beta none."""
    (tmp_path / "Snakefile").write_text(SNAKEFILE)
    output_root = tmp_path / "extra/output"
    output_root.mkdir(parents=True)
    job_record_examples.write_job_record(
        output_root / "alpha.job_record.json",
        job_record_examples.job_record(
            JobIdentity(stage="training", name="alpha", dataset_key="cube"),
            started_at="2026-10-08T12:00:00+00:00",
            ended_at="2026-10-08T14:00:00+00:00",
            job_duration_seconds=7200,
            cpu_model="AMD EPYC 7742",
            host="node0",
        ),
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def estimate(*options: str) -> Result:
    return CliRunner().invoke(app, [*options, "--cores", "2", "all"])


@pytest.mark.usefixtures("workflow")
def test_strict_fails_when_a_planned_job_is_unestimated() -> None:
    lenient = estimate()
    strict = estimate("--strict")

    assert lenient.exit_code == 0, lenient.output
    assert "1 exact, 0 pooled, 1 unestimated" in lenient.output
    assert strict.exit_code == 1
    assert "1 planned job is unestimated" in strict.output


def test_the_per_job_csv_holds_each_job_estimate(workflow: Path) -> None:
    result = estimate("--output", "estimate.csv", "--by", "name")

    assert result.exit_code == 0, result.output
    with (workflow / "estimate.csv").open(newline="") as stream:
        rows = {row["name"]: row for row in csv.DictReader(stream)}
    assert rows["alpha"]["match_level"] == "exact"
    # 2 hours on the 2 CPUs the rule's threads request
    assert float(rows["alpha"]["mean_core_hours"]) == 4
    assert rows["beta"]["match_level"] == "unestimated"
    assert "Totals by name" in result.output


def test_job_durations_can_come_from_a_job_duration_table(
    workflow: Path,
) -> None:
    table = workflow / "release/job_duration_table.parquet"
    write_job_duration_table(workflow / "extra/output", table)
    shutil.rmtree(workflow / "extra/output")

    result = estimate("--job-durations", str(table))

    assert result.exit_code == 0, result.output
    assert f"Job durations from {table}: 1 job record matched" in result.output
    assert "1 exact" in result.output


@pytest.mark.usefixtures("workflow")
def test_a_missing_job_duration_source_is_refused() -> None:
    result = estimate("--job-durations", "missing.parquet")

    assert result.exit_code != 0
    assert "missing.parquet" in result.output


def test_without_an_output_root_every_job_is_unestimated(
    workflow: Path,
) -> None:
    shutil.rmtree(workflow / "extra/output")

    result = estimate()

    assert result.exit_code == 0, result.output
    assert "0 exact, 0 pooled, 2 unestimated" in result.output


def test_a_restored_release_table_is_named_by_its_release(
    workflow: Path,
) -> None:
    write_catalog(workflow / "source", Catalog(methods=[ALPHA]))
    write_job_record(workflow / "source", ALPHA_METHOD_RECORD)
    assert save(workflow).exit_code == 0
    assert restore(workflow).exit_code == 0
    table = workflow / "checkout/extra/release_job_duration_table.parquet"

    result = estimate("--job-durations", str(table))

    assert result.exit_code == 0, result.output
    assert (
        f"Job durations from {table} (release 2026-10-cube, partial scope)"
        in result.output
    )
    assert "0 exact, 1 pooled, 1 unestimated" in result.output


def test_a_table_left_from_another_release_is_not_named_by_the_manifest(
    workflow: Path,
) -> None:
    write_catalog(workflow / "source", Catalog(methods=[ALPHA]))
    write_job_record(workflow / "source", ALPHA_METHOD_RECORD)
    assert save(workflow).exit_code == 0
    assert restore(workflow).exit_code == 0
    table = workflow / "checkout/extra/release_job_duration_table.parquet"
    # As a download of another release with --overwrite leaves it
    write_job_duration_table(workflow / "extra/output", table)

    result = estimate("--job-durations", str(table))

    assert result.exit_code == 0, result.output
    assert "(not the table of release 2026-10-cube" in result.output
