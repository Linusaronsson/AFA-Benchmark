import json
from pathlib import Path

import pytest

from scripts.analysis.collect_compute import (
    benchmark_identity,
    namespace_gpu_telemetry,
    namespace_provenance,
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


def test_namespace_provenance_rejects_mixed_hardware(tmp_path: Path) -> None:
    root = tmp_path / "manifests" / "study"
    root.mkdir(parents=True)
    (root / "1.json").write_text(json.dumps(_manifest("cuda")))
    (root / "2.json").write_text(json.dumps(_manifest("cpu")))
    with pytest.raises(ValueError, match="mixes execution environments"):
        namespace_provenance(tmp_path / "manifests", "study")


def test_namespace_provenance_rejects_mixed_mps_modes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "manifests" / "study"
    root.mkdir(parents=True)
    (root / "1.json").write_text(json.dumps(_manifest("cuda")))
    (root / "2.json").write_text(json.dumps(_manifest("cuda", mps=True)))

    with pytest.raises(ValueError, match="mixes execution environments"):
        namespace_provenance(tmp_path / "manifests", "study")


def test_namespace_provenance_marks_every_missing_field_untracked(
    tmp_path: Path,
) -> None:
    provenance = namespace_provenance(tmp_path / "manifests", "study")

    assert provenance == {
        "hardware_signature": "untracked",
        "git_commit": "untracked",
        "device": "untracked",
        "cores": "untracked",
        "mem_mb": "untracked",
        "gpu_workers": "untracked",
        "mps": "untracked",
        "architecture": "untracked",
        "torch": "untracked",
        "torch_cuda": "untracked",
        "cuda_devices": "[]",
    }


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
