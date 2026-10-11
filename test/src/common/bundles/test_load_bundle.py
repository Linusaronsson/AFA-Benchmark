"""`load_bundle` returns the object with its typed manifest."""

import json
from pathlib import Path

import pytest
from torch import nn

from afabench.core.bundle_system.bundle import (
    load_bundle,
    read_manifest,
    save_bundle,
)
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.core.provenance import ProvenanceInput
from afabench.core.types import AFADataset
from afabench.datasets.datasets import CubeDataset
from afabench.testing.provenance import placeholder_provenance

MODEL_RECORD = placeholder_provenance(
    "pretraining", dataset_realization_index=2
)


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    path = tmp_path / "model.bundle"
    save_bundle(
        TorchModelBundle(nn.Linear(2, 2)),
        path,
        metadata={"free": "form"},
        provenance=MODEL_RECORD,
    )
    return path


def test_loaded_bundle_carries_its_typed_manifest(model_path: Path) -> None:
    loaded = load_bundle(model_path)

    assert isinstance(loaded.obj, TorchModelBundle)
    assert loaded.manifest.class_name == "TorchModelBundle"
    assert loaded.manifest.metadata == {"free": "form"}
    assert loaded.manifest.provenance == MODEL_RECORD
    assert (
        loaded.manifest.content_hash
        == read_manifest(model_path)["content_hash"]
    )


def test_loaded_bundle_is_an_input_recorded_with_the_path_as_given(
    model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(model_path.parent)

    loaded = load_bundle("./model.bundle")

    assert loaded.as_input("pretrained_model") == ProvenanceInput(
        role="pretrained_model",
        path="./model.bundle",
        class_name="TorchModelBundle",
        content_hash=read_manifest(model_path)["content_hash"],
    )


def test_expect_returns_an_object_of_the_expected_class(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.bundle"
    save_bundle(
        CubeDataset(n_samples=4, seed=0),
        path,
        metadata={},
        provenance=placeholder_provenance(),
    )

    dataset = load_bundle(path, expect=AFADataset).obj

    assert len(dataset) == 4


def test_expect_rejects_an_object_of_another_class(model_path: Path) -> None:
    with pytest.raises(
        TypeError, match=r"Expected CubeDataset in .*model\.bundle, got "
    ):
        load_bundle(model_path, expect=CubeDataset)


def test_legacy_manifest_has_no_provenance_or_content_hash(
    model_path: Path,
) -> None:
    manifest = read_manifest(model_path)
    del manifest["provenance"]
    del manifest["content_hash"]
    manifest["bundle_version"] = "1.0.0"
    (model_path / "manifest.json").write_text(json.dumps(manifest))

    loaded = load_bundle(model_path)

    assert loaded.manifest.provenance is None
    assert loaded.manifest.content_hash is None
