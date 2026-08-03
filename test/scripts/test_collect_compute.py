import json
from pathlib import Path

import pytest

from scripts.analysis.collect_compute import (
    benchmark_identity,
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


def _manifest(device: str) -> dict[str, object]:
    return {
        "git_commit": "abc123",
        "command": {"device": device},
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
