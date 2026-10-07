"""Classifier training records provenance in the classifier bundle (ADR 0002)."""

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from afabench.core.bundle_system.bundle import (
    bundle_provenance,
    load_bundle,
    read_manifest,
    save_bundle,
)
from afabench.core.provenance import ProvenanceInput
from afabench.datasets.datasets import CubeDataset
from afabench.testing.provenance import placeholder_provenance
from scripts.train_classifier.masked_mlp_classifier import main

REPO_ROOT = Path(__file__).parents[2]
CONFIG_DIR = (
    REPO_ROOT / "extra/conf/scripts/train_classifier/masked_mlp_classifier"
)
SEED = 5


@pytest.fixture
def dataset_bundles(tmp_path: Path) -> tuple[Path, Path]:
    train_path = tmp_path / "train.bundle"
    val_path = tmp_path / "val.bundle"
    save_bundle(
        CubeDataset(n_samples=64, seed=0),
        train_path,
        metadata={},
        provenance=placeholder_provenance(
            dataset_key="cube", dataset_realization_index=4, split="train"
        ),
    )
    save_bundle(
        CubeDataset(n_samples=32, seed=1),
        val_path,
        metadata={},
        provenance=placeholder_provenance(
            dataset_key="cube", dataset_realization_index=4, split="val"
        ),
    )
    return train_path, val_path


@pytest.fixture
def restore_matmul_precision() -> Iterator[None]:
    """Undo the script's process-wide matmul precision after the test."""
    precision = torch.get_float32_matmul_precision()
    yield
    torch.set_float32_matmul_precision(precision)


def _overrides(train_path: Path, val_path: Path, save_path: Path) -> list[str]:
    return [
        f"train_dataset_path={train_path}",
        f"val_dataset_path={val_path}",
        f"save_path={save_path}",
        "initializer=cold",
        "unmasker=direct",
        "device=cpu",
        f"seed={SEED}",
        "use_wandb=False",
        "smoke_test=True",
        "experiment@_global_=cube",
        "epochs=1",
        "num_cells=[4]",
    ]


def _assert_records_inputs_seed_and_dataset(
    save_path: Path, train_path: Path, val_path: Path
) -> None:
    assert load_bundle(save_path, device=torch.device("cpu"))[0] is not None
    record = bundle_provenance(save_path)
    assert record is not None
    assert record.stage == "classifier_training"
    assert record.seed == SEED
    assert record.smoke_test is True
    assert record.method_name is None
    assert record.split is None
    assert (record.dataset_key, record.dataset_realization_index) == (
        "cube",
        4,
    )
    assert record.inputs == [
        ProvenanceInput(
            role="train_dataset",
            path=str(train_path),
            class_name="CubeDataset",
            content_hash=read_manifest(train_path)["content_hash"],
        ),
        ProvenanceInput(
            role="val_dataset",
            path=str(val_path),
            class_name="CubeDataset",
            content_hash=read_manifest(val_path)["content_hash"],
        ),
    ]
    # The config as the script used it, after its smoke-test overrides
    assert record.resolved_config["num_cells"] == [4]
    assert record.resolved_config["batch_size"] == 32
    assert record.resolved_config["smoke_test"] is True
    assert record.compute.device == "cpu"
    assert record.compute.float32_matmul_precision == "medium"


@pytest.mark.usefixtures("restore_matmul_precision")
def test_masked_mlp_classifier_bundle_records_its_inputs_seed_and_dataset(
    dataset_bundles: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_path, val_path = dataset_bundles
    save_path = tmp_path / "classifier.bundle"
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=_overrides(train_path, val_path, save_path),
        )
    # Lightning writes its logs relative to the working directory
    monkeypatch.chdir(tmp_path)

    # Hydra passes a given config straight to the script, in this process
    main(cfg)

    _assert_records_inputs_seed_and_dataset(save_path, train_path, val_path)


# A Hydra subprocess takes about 10 s, too slow for the default suite
@pytest.mark.pipeline
def test_masked_mlp_classifier_script_records_its_inputs_seed_and_dataset(
    dataset_bundles: tuple[Path, Path], tmp_path: Path
) -> None:
    train_path, val_path = dataset_bundles
    save_path = tmp_path / "classifier.bundle"
    command = [
        sys.executable,
        "scripts/train_classifier/masked_mlp_classifier.py",
        *_overrides(train_path, val_path, save_path),
        f"hydra.run.dir={tmp_path / 'hydra'}",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    _assert_records_inputs_seed_and_dataset(save_path, train_path, val_path)
