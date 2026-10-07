"""Bundles carry a provenance record (ADR 0002, required tests 1 to 3)."""

import json
from pathlib import Path

import pytest
from torch import nn

from afabench.core.bundle_system.bundle import (
    bundle_input,
    bundle_provenance,
    compute_content_hash,
    load_bundle,
    read_manifest,
    save_bundle,
)
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.core.provenance import (
    ProvenanceInput,
    ProvenanceRecord,
    capture_provenance,
)
from afabench.datasets.datasets import CubeDataset


def _full_record() -> ProvenanceRecord:
    return capture_provenance(
        stage="training",
        resolved_config={
            "seed": 7,
            "hard_budget": None,
            "initializer": {"class_name": "ZeroInitializer", "kwargs": {}},
            "layers": [16, 8],
            "smoke_test": False,
        },
        seed=7,
        smoke_test=False,
        device="cpu",
        inputs=[
            ProvenanceInput(
                role="train_dataset",
                path="datasets/cube/0/train.bundle",
                class_name="CubeDataset",
                content_hash="sha256:" + "a" * 64,
            ),
            ProvenanceInput(
                role="classifier",
                path="classifier.bundle",
                class_name="WrappedMaskedMLPClassifier",
                content_hash=None,
            ),
            ProvenanceInput(
                role="pretrained_model",
                path="model.bundle",
                class_name="TorchModelBundle",
                content_hash="sha256:" + "b" * 64,
            ),
        ],
        method_name="gdfs",
        dataset_key="cube",
        dataset_realization_index=0,
        split=None,
    )


def _dataset_record(split: str = "train") -> ProvenanceRecord:
    return capture_provenance(
        stage="dataset_generation",
        resolved_config={"n_samples": 10},
        seed=1,
        smoke_test=False,
        device="cpu",
        dataset_key="cube",
        dataset_realization_index=0,
        split=split,  # pyright: ignore[reportArgumentType]
    )


def test_record_round_trips_through_the_bundle(tmp_path: Path) -> None:
    record = _full_record()
    path = tmp_path / "method.bundle"

    save_bundle(
        TorchModelBundle(nn.Linear(2, 2)),
        path,
        metadata={"free": "form"},
        provenance=record,
    )
    _, manifest = load_bundle(path)

    assert manifest["bundle_version"] == "1.1.0"
    assert manifest["metadata"] == {"free": "form"}
    assert ProvenanceRecord.from_json_dict(manifest["provenance"]) == record
    assert bundle_provenance(path) == record


def test_legacy_manifest_without_provenance_still_loads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.bundle"
    save_bundle(
        TorchModelBundle(nn.Linear(2, 2)),
        path,
        metadata={"seed": 0},
        provenance=_full_record(),
    )
    manifest = read_manifest(path)
    del manifest["provenance"]
    del manifest["content_hash"]
    manifest["bundle_version"] = "1.0.0"
    (path / "manifest.json").write_text(json.dumps(manifest))

    loaded, loaded_manifest = load_bundle(path)

    assert isinstance(loaded, TorchModelBundle)
    assert loaded_manifest["metadata"] == {"seed": 0}
    assert bundle_provenance(path) is None
    assert bundle_input("classifier", str(path)) == ProvenanceInput(
        role="classifier",
        path=str(path),
        class_name="TorchModelBundle",
        content_hash=None,
    )


def test_save_bundle_requires_a_provenance_record(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="provenance"):
        save_bundle(  # pyright: ignore[reportCallIssue]
            TorchModelBundle(nn.Linear(2, 2)),
            tmp_path / "x.bundle",
            metadata={},
        )


def test_identical_data_gives_the_same_content_hash(tmp_path: Path) -> None:
    dataset = CubeDataset(n_samples=12, seed=3)
    first = tmp_path / "first.bundle"
    second = tmp_path / "second.bundle"

    save_bundle(
        dataset, first, metadata={"a": 1}, provenance=_dataset_record()
    )
    save_bundle(
        dataset, second, metadata={"b": 2}, provenance=_dataset_record("val")
    )

    first_hash = read_manifest(first)["content_hash"]
    assert first_hash == read_manifest(second)["content_hash"]
    assert first_hash.startswith("sha256:")
    assert len(first_hash) == len("sha256:") + 64
    assert first_hash == compute_content_hash(first / "data")


def test_changing_one_data_file_changes_the_content_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.bundle"
    save_bundle(
        CubeDataset(n_samples=12, seed=3),
        path,
        metadata={},
        provenance=_dataset_record(),
    )
    recorded = read_manifest(path)["content_hash"]
    data_file = next(file for file in (path / "data").iterdir())
    data_file.write_bytes(data_file.read_bytes() + b"\0")

    assert compute_content_hash(path / "data") != recorded


def test_consumer_input_carries_the_producers_content_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.bundle"
    save_bundle(
        CubeDataset(n_samples=12, seed=3),
        path,
        metadata={},
        provenance=_dataset_record(),
    )

    entry = bundle_input("train_dataset", str(path))

    assert entry == ProvenanceInput(
        role="train_dataset",
        path=str(path),
        class_name="CubeDataset",
        content_hash=read_manifest(path)["content_hash"],
    )
    assert entry.content_hash is not None
