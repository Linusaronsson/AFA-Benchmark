"""Adding dataset realizations to a run through the real orchestration."""

from pathlib import Path

from test.workflow.test_cpu_processing_execution import processing_workflow
from test.workflow.test_full_reproduction import planned_jobs


def test_extending_realizations_plans_only_the_new_ones(
    tmp_path: Path,
) -> None:
    workflow = processing_workflow(tmp_path)
    workflow.config["dataset_realization_indices"] = [0, 1]
    first = workflow.run(target="all_train_classifiers")
    assert first.returncode == 0, first.stdout + first.stderr

    workflow.config["dataset_realization_indices"] = [0, 1, 2]
    result = workflow.run("--dry-run", target="all_train_classifiers")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    jobs = [
        job
        for job in planned_jobs(output)
        if job["rule"] != "all_train_classifiers"
    ]
    assert sorted((job["rule"], job.get("wildcards", "")) for job in jobs) == [
        ("dataset_generation", "dataset=cube, dataset_realization_index=2"),
        ("train_classifier", "dataset=cube, dataset_realization_index=2"),
    ]


def test_realizations_generated_together_are_not_regenerated(
    tmp_path: Path,
) -> None:
    workflow = processing_workflow(tmp_path)
    workflow.config["dataset_realization_indices"] = [0, 1]
    first = workflow.run(target="all_train_classifiers")
    assert first.returncode == 0, first.stdout + first.stderr
    # The layout before one job per realization: one record per dataset key
    datasets = workflow.output / "datasets/cube"
    for record in datasets.glob("*/dataset_generation.job_record.json"):
        record.unlink()
    (datasets / "dataset_generation.job_record.json").write_text("{}")

    result = workflow.run("--dry-run", target="all_train_classifiers")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert planned_jobs(output) == []
