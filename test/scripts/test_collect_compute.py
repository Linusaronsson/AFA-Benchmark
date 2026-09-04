import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.collect_compute import (
    attribute_runs,
    benchmark_identity,
    filter_methods,
    namespace_gpu_telemetry,
    namespace_runs,
    validate_unique_records,
)


def test_benchmark_identity_parses_current_train_path() -> None:
    root = Path("benchmark")
    path = (
        root
        / "train_method"
        / "method-ol_with_mask"
        / (
            "dataset-cube_nm+mechanism-mcar+p-0.5+strategy-restricted+"
            "instance-3+train_hard_budget-7.tsv"
        )
    )
    identity = benchmark_identity(path, root, "train_method")
    assert identity == {
        "rule": "train_method",
        "method": "ol_with_mask",
        "dataset": "cube_nm",
        "mechanism": "mcar",
        "p": 0.5,
        "strategy": "restricted",
        "instance": 3,
        "train_hard_budget": 7.0,
    }


def test_benchmark_identity_preserves_decimal_eval_coordinates() -> None:
    root = Path("eval")
    path = (
        root
        / "dataset-diabetes"
        / (
            "method-aaco+mechanism-mnar_self+p-0.5+strategy-pvae_stepwise+"
            "instance-3+train_hard_budget-20+eval_hard_budget-14"
        )
        / "benchmark.tsv"
    )

    identity = benchmark_identity(path, root, "eval_method")

    assert identity == {
        "rule": "eval_method",
        "dataset": "diabetes",
        "method": "aaco",
        "mechanism": "mnar_self",
        "p": 0.5,
        "strategy": "pvae_stepwise",
        "instance": 3,
        "train_hard_budget": 20.0,
        "eval_hard_budget": 14.0,
    }


def _manifest(device: str, *, mps: bool = False) -> dict[str, object]:
    return {
        "git_commit": "abc123",
        "command": {
            "device": device,
            "cores": 16,
            "mem_mb": 115_200,
            "gpu_workers": 4,
            "mps": mps,
        },
        "host": {"architecture": "aarch64"},
        "software": {"torch": "2.11.0", "torch_cuda": "13.0"},
        "cuda_devices": ["NVIDIA GH200 120GB"],
    }


def _write_runs(root: Path, *, created_at_suffix: str = "+00:00") -> None:
    """
    Write runs in the shape production writes them.

    `run_manifest.py` stamps `created_at` from an aware `datetime`, so the
    default suffix is the offset a real manifest carries. A file mtime is bare
    epoch seconds, and ordering the two requires both sides in UTC.
    """
    root.mkdir(parents=True)
    first = _manifest("cuda")
    first["created_at"] = f"2026-08-01T00:00:00{created_at_suffix}"
    first["git_commit"] = "aaaa"
    second = _manifest("cuda")
    second["created_at"] = f"2026-08-05T00:00:00{created_at_suffix}"
    second["git_commit"] = "bbbb"
    (root / "1.json").write_text(json.dumps(first))
    (root / "2.json").write_text(json.dumps(second))


def test_attribute_runs_gives_each_benchmark_its_own_run(
    tmp_path: Path,
) -> None:
    """A namespace may hold several runs; each cell keeps the one that made it."""
    _write_runs(tmp_path / "manifests" / "study")
    early, late = tmp_path / "early.tsv", tmp_path / "late.tsv"
    early.write_text("x")
    late.write_text("x")
    os.utime(early, (0, datetime(2026, 8, 2, tzinfo=UTC).timestamp()))
    os.utime(late, (0, datetime(2026, 8, 6, tzinfo=UTC).timestamp()))
    frame = pd.DataFrame({"path": [str(early), str(late)]})

    attributed = attribute_runs(
        frame, namespace_runs(tmp_path / "manifests", "study")
    )

    assert list(attributed["git_commit"]) == ["aaaa", "bbbb"]


def test_attribute_runs_accepts_a_manifest_without_an_offset(
    tmp_path: Path,
) -> None:
    """A naive `created_at` is read as UTC rather than refusing to order."""
    _write_runs(tmp_path / "manifests" / "study", created_at_suffix="")
    path = tmp_path / "late.tsv"
    path.write_text("x")
    os.utime(path, (0, datetime(2026, 8, 6, tzinfo=UTC).timestamp()))

    attributed = attribute_runs(
        pd.DataFrame({"path": [str(path)]}),
        namespace_runs(tmp_path / "manifests", "study"),
    )

    assert list(attributed["git_commit"]) == ["bbbb"]


def test_attribute_runs_assigns_a_pre_manifest_benchmark_to_the_first_run(
    tmp_path: Path,
) -> None:
    _write_runs(tmp_path / "manifests" / "study")
    path = tmp_path / "old.tsv"
    path.write_text("x")
    os.utime(path, (0, datetime(2026, 7, 1, tzinfo=UTC).timestamp()))

    attributed = attribute_runs(
        pd.DataFrame({"path": [str(path)]}),
        namespace_runs(tmp_path / "manifests", "study"),
    )

    assert list(attributed["git_commit"]) == ["aaaa"]


def test_attribute_runs_marks_every_field_untracked_without_manifests(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a.tsv"
    path.write_text("x")

    attributed = attribute_runs(
        pd.DataFrame({"path": [str(path)]}),
        namespace_runs(tmp_path / "manifests", "study"),
    )

    assert attributed["hardware_signature"].tolist() == ["untracked"]
    assert attributed["git_commit"].tolist() == ["untracked"]


def test_namespace_gpu_telemetry_summarizes_active_samples(
    tmp_path: Path,
) -> None:
    root = tmp_path / "telemetry" / "study"
    root.mkdir(parents=True)
    (root / "1.csv").write_text(
        "timestamp,gpu_index,utilization_percent,memory_used_mb,"
        "memory_total_mb,power_draw_w,temperature_c\n"
        "2026/08/03 10:00:00,0,0,100,1000,50,30\n"
        "2026/08/03 10:00:05,0,20,200,1000,60,31\n"
        "2026/08/03 10:00:10,0,80,400,1000,90,32\n"
    )

    summary = namespace_gpu_telemetry(tmp_path / "telemetry", "study")

    assert summary["gpu_samples"] == 3
    assert summary["gpu_active_samples"] == 2
    assert summary["gpu_utilization_mean_percent"] == pytest.approx(100 / 3)
    assert summary["gpu_utilization_active_median_percent"] == 50
    assert summary["gpu_memory_peak_mb"] == 400
    assert summary["gpu_power_peak_w"] == 90


def test_filter_methods_excludes_shared_archive_records() -> None:
    frame = pd.DataFrame(
        [
            {"rule": "train_method", "method": "gdfs"},
            {"rule": "eval_method", "method": "jafa"},
            {"rule": "pretrain_method", "pretrain_key": "gdfs"},
            {"rule": "restore_view"},
            {"rule": "train_method", "method": "aaco"},
        ]
    )

    selected = filter_methods(frame, {"gdfs", "jafa"})

    assert selected["rule"].tolist() == [
        "train_method",
        "eval_method",
        "pretrain_method",
    ]


def test_validate_unique_records_rejects_a_repeated_scientific_cell() -> None:
    row = {
        "namespace": "study",
        "rule": "train_method",
        "dataset": "cube",
        "mechanism": "mcar",
        "p": 0.5,
        "strategy": "restricted",
        "instance": 0,
        "method": "gdfs",
    }

    with pytest.raises(ValueError, match="duplicate compute records"):
        validate_unique_records(pd.DataFrame([row, row]))
