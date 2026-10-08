"""Job records the computational stages leave beside their artifacts."""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from test.workflow.submission_harness import REPO_ROOT
from test.workflow.test_full_reproduction import (
    GPU_JOBS,
    full_benchmark_workflow,
)

TAG = "initializer-cold"
REALIZATIONS = [0, 1]
# Pretrained model of each method in full_benchmark_workflow, None for none.
PRETRAINED_MODELS = {
    "alpha": "shared",
    "beta": "shared",
    "gamma": "other",
    "delta": None,
}
# CPUs this test's --set-resources gives pretraining and evaluation. It
# replaces the mixed-gres profile's set-resources, so the other jobs get the
# profile's default-resources.
PRETRAINING_CPUS = 3
EVALUATION_CPUS = 4
DEFAULT_CPUS = 1
UNIDENTIFIED = {
    "name": None,
    "dataset_realization_index": None,
    "pretrain_seed": None,
    "train_seed": None,
    "eval_seed": None,
    "train_hard_budget": None,
    "train_soft_budget_param": None,
    "eval_hard_budget": None,
    "eval_soft_budget_param": None,
    "eval_batch_size": None,
}


def allocation(*, gpu: bool, cpus: int) -> dict[str, Any]:
    return {
        "device": "cuda" if gpu else "cpu",
        "cpus": cpus,
        "gpus": 1 if gpu else 0,
    }


def expected_records() -> dict[str, dict[str, Any]]:
    """Every job record of the run, by path, with identity and allocation."""
    records: dict[str, dict[str, Any]] = {
        "datasets/cube.job_record.json": {
            **UNIDENTIFIED,
            "stage": "dataset_generation",
            "dataset_key": "cube",
            **allocation(gpu=False, cpus=DEFAULT_CPUS),
        }
    }
    for k in REALIZATIONS:
        realization = f"dataset-cube+realization_index-{k}"
        identity = {
            **UNIDENTIFIED,
            "dataset_key": "cube",
            "dataset_realization_index": k,
        }
        for owner, classifier, gpu in [
            ("", "masked_mlp_classifier", True),
            ("method-alpha+", "special", False),
        ]:
            records[
                f"trained_classifiers/{TAG}/{owner}{realization}"
                ".job_record.json"
            ] = {
                **identity,
                "stage": "classifier_training",
                "name": classifier,
                "train_seed": k,
                **allocation(gpu=gpu, cpus=DEFAULT_CPUS),
            }
        for pretrained_model in ["shared", "other"]:
            records[
                f"pretrained_models/{TAG}/{pretrained_model}/{realization}/"
                f"pretrain_seed-{k}/model.job_record.json"
            ] = {
                **identity,
                "stage": "pretraining",
                "name": pretrained_model,
                "pretrain_seed": k,
                **allocation(
                    gpu=pretrained_model == "shared", cpus=PRETRAINING_CPUS
                ),
            }
        for method, pretrained_model in PRETRAINED_MODELS.items():
            pretrain_folder = (
                "NO_PRETRAIN"
                if pretrained_model is None
                else f"pretrain_seed-{k}"
            )
            training = (
                f"{TAG}/{method}/{realization}/{pretrain_folder}/"
                f"train_seed-{k}+train_hard_budget-1+"
                "train_soft_budget_param-null"
            )
            evaluation = (
                f"{training}/eval_seed-{k}+eval_hard_budget-1+"
                "eval_soft_budget_param-null"
            )
            training_identity = {
                **identity,
                "name": method,
                "pretrain_seed": None if pretrained_model is None else k,
                "train_seed": k,
                "train_hard_budget": 1,
            }
            evaluation_identity = {
                **training_identity,
                "eval_seed": k,
                "eval_hard_budget": 1,
            }
            records[f"trained_methods/{training}/method.job_record.json"] = {
                **training_identity,
                "stage": "training",
                **allocation(
                    gpu=("train_method", method) in GPU_JOBS,
                    cpus=DEFAULT_CPUS,
                ),
            }
            records[
                f"eval_results/eval_split-test/{evaluation}/"
                "eval_data.job_record.json"
            ] = {
                **evaluation_identity,
                "stage": "evaluation",
                "eval_batch_size": 1,
                **allocation(
                    gpu=("eval_method", method) in GPU_JOBS,
                    cpus=EVALUATION_CPUS,
                ),
            }
            records[
                f"eval_results_transformed/eval_split-test/{evaluation}/"
                "eval_data.job_record.json"
            ] = {
                **evaluation_identity,
                "stage": "transformation",
                **allocation(gpu=False, cpus=DEFAULT_CPUS),
            }
    return records


@dataclass
class RecordedRun:
    output: Path
    records: dict[str, dict[str, Any]]


@pytest.fixture(scope="module")
def recorded_run(tmp_path_factory: pytest.TempPathFactory) -> RecordedRun:
    root = tmp_path_factory.mktemp("job-records")
    workflow = full_benchmark_workflow(root)
    # The site profile's allocation, run locally: the fake SLURM service
    # takes minutes for this graph.
    result = workflow.run(
        "--workflow-profile",
        str(root / "extra/workflow/profiles/mixed-gres"),
        "--executor",
        "local",
        "--set-resources",
        f"pretrain_model:cpus_per_task={PRETRAINING_CPUS}",
        f"eval_method:cpus_per_task={EVALUATION_CPUS}",
        target="all",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = root / "extra/output"
    return RecordedRun(
        output,
        {
            path.relative_to(output).as_posix(): json.loads(path.read_text())
            for path in output.rglob("*.job_record.json")
        },
    )


def test_every_computational_job_leaves_its_identity_and_allocation(
    recorded_run: RecordedRun,
) -> None:
    expected = expected_records()

    assert {
        path: {field: record.get(field) for field in expected[path]}
        for path, record in recorded_run.records.items()
        if path in expected
    } == expected
    assert set(recorded_run.records) == set(expected)


def test_job_records_sit_beside_their_artifacts(
    recorded_run: RecordedRun,
) -> None:
    # A record is named after its artifact: model.bundle, model.job_record.json
    for path in recorded_run.records:
        record = recorded_run.output / path
        artifact_name = record.name.removesuffix(".job_record.json")
        artifacts = [
            sibling
            for sibling in record.parent.iterdir()
            if sibling != record
            and sibling.name.split(".")[0] == artifact_name
        ]
        assert len(artifacts) == 1, path


def test_job_records_carry_timing_code_commit_and_smoke_flag(
    recorded_run: RecordedRun,
) -> None:
    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    for path, record in recorded_run.records.items():
        started = datetime.fromisoformat(record["started_at"])
        ended = datetime.fromisoformat(record["ended_at"])
        assert started.utcoffset() is not None, path
        assert started <= ended, path
        assert record["job_duration_seconds"] == pytest.approx(
            (ended - started).total_seconds(), abs=1e-3
        )
        assert record["job_record_version"] == 1, path
        assert record["exit_status"] == "completed", path
        assert record["exit_code"] == 0, path
        assert record["smoke_test"] is True, path
        assert record["code_commit"] == commit, path
        for field in ["cpu_model", "host"]:
            assert isinstance(record[field], str), (path, field)
        assert "gpu_model" in record, path
        assert "slurm_job_id" in record, path


def test_time_files_and_time_aggregation_still_work(
    recorded_run: RecordedRun,
) -> None:
    for name, count in [
        ("pretrain_time.txt", 4),
        ("train_time.txt", 8),
        ("eval_time.txt", 8),
    ]:
        paths = list(recorded_run.output.rglob(name))
        assert len(paths) == count, name
        for path in paths:
            assert float(path.read_text()) >= 0, path
    time_plots = (
        recorded_run.output / f"plot_results/eval_split-test/{TAG}/time"
    )
    assert (time_plots / "fixture.svg").is_file()
