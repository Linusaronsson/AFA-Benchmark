"""Release reuse verifies bytes and stays scoped to released consumers."""

import json
import shutil
from pathlib import Path

import pytest

from afabench.core.output_layout import OutputLayout
from afabench.release.reuse import ReleaseReuse
from test.scripts.release_artifacts import (
    ALPHA,
    HARD_3,
    Catalog,
    classifier_bundle,
    dataset_bundle,
    evaluation_table,
    method_bundle,
)
from test.scripts.test_release_manifest import save_catalog


@pytest.fixture
def released(tmp_path: Path) -> Path:
    save_catalog(
        tmp_path,
        Catalog(methods=[ALPHA], datasets=["cube", "cube_without_noise"]),
    )
    root = tmp_path / "source"
    shutil.copy2(tmp_path / "snapshot/release_manifest.json", root)
    return root


@pytest.mark.parametrize("kind", ["classifier", "method", "evaluation"])
def test_compatible_release_consumers(released: Path, kind: str) -> None:
    if kind == "classifier":
        consumer = classifier_bundle("cube", 0)
        role, split = "train_dataset", "train"
    elif kind == "method":
        consumer = method_bundle(ALPHA, "cube", 0, HARD_3)
        role, split = "val_dataset", "val"
    else:
        consumer = "eval_results/" + evaluation_table(ALPHA, "cube", 0, HARD_3)
        role, split = "eval_dataset", "test"
    assert ReleaseReuse(released).compatible(
        released / consumer, released / dataset_bundle("cube", 0, split), role
    )


def test_changed_dataset_bytes_fail_even_with_unchanged_manifest(
    released: Path,
) -> None:
    dataset = released / dataset_bundle("cube", 0, "train")
    (dataset / "data/weights.bin").write_text("changed")
    reuse = ReleaseReuse(released)
    layout = OutputLayout(
        root=str(released), initializer="cold", eval_split="test"
    )
    # A changed dataset outside the invocation is not examined.
    reuse.validate_datasets(layout, ["cube_without_noise"], [0])
    with pytest.raises(ValueError, match=r"Incompatible dataset.*cube"):
        reuse.validate_datasets(layout, ["cube"], [0])


@pytest.mark.parametrize("replacement", ["missing", "local"])
def test_missing_or_locally_replaced_models_keep_normal_dependencies(
    released: Path, replacement: str
) -> None:
    consumer = released / method_bundle(ALPHA, "cube", 0, HARD_3)
    if replacement == "missing":
        shutil.rmtree(consumer)
    else:
        manifest = consumer / "manifest.json"
        data = json.loads(manifest.read_text())
        data["content_hash"] = "sha256:local"
        manifest.write_text(json.dumps(data))
    assert not ReleaseReuse(released).compatible(
        consumer,
        released / dataset_bundle("cube", 0, "train"),
        "train_dataset",
    )


def test_unknown_input_hash_is_not_compatible(released: Path) -> None:
    manifest = released / "release_manifest.json"
    data = json.loads(manifest.read_text())
    consumer = method_bundle(ALPHA, "cube", 0, HARD_3)
    for bundle in data["bundles"]:
        if bundle["path"] == consumer:
            bundle["inputs"][0]["content_hash"] = None
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incompatible dataset"):
        ReleaseReuse(released).compatible(
            released / consumer,
            released / dataset_bundle("cube", 0, "train"),
            "train_dataset",
        )


def test_missing_dataset_requires_generation_first(released: Path) -> None:
    dataset = released / dataset_bundle("cube", 0, "train")
    shutil.rmtree(dataset)
    with pytest.raises(ValueError, match="all_generate_datasets first"):
        ReleaseReuse(released).compatible(
            released / classifier_bundle("cube", 0), dataset, "train_dataset"
        )
