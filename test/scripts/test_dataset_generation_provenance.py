"""Dataset generation records provenance in every split bundle (ADR 0002)."""

import subprocess
import sys
from pathlib import Path

import pytest

from afabench.core.bundle_system.bundle import bundle_provenance
from afabench.datasets.config import SplitRatioConfig
from afabench.datasets.datasets import CubeDataset
from scripts.dataset_generation.generate_dataset import generate_and_save_split

REPO_ROOT = Path(__file__).parents[2]
SPLITS = ("train", "val", "test")


def test_generated_split_bundles_record_their_identity_and_seed(
    tmp_path: Path,
) -> None:
    resolved_config = {
        "save_path": str(tmp_path),
        "dataset_realization_indices": [3],
        "seeds": [11],
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "dataset": {
            "class_name": "CubeDataset",
            "kwargs": {"n_samples": 20},
        },
    }

    generate_and_save_split(
        dataset_class=CubeDataset,
        split_ratio=SplitRatioConfig(train=0.6, val=0.2, test=0.2),
        seed_for_split=11,
        save_path=tmp_path / "3",
        dataset_kwargs={"n_samples": 20, "seed": 11},
        metadata_to_save={"dataset_realization_index": 3},
        dataset_key="cube_without_noise",
        dataset_realization_index=3,
        resolved_config=resolved_config,
    )

    for split in SPLITS:
        record = bundle_provenance(tmp_path / "3" / f"{split}.bundle")
        assert record is not None, split
        assert record.stage == "dataset_generation"
        assert record.split == split
        assert record.dataset_key == "cube_without_noise"
        assert record.dataset_realization_index == 3
        assert record.seed == 11
        assert record.smoke_test is False
        assert record.method_name is None
        assert record.inputs == []
        assert record.resolved_config == resolved_config


# A Hydra subprocess takes about 7 s, too slow for the default suite;
# generate_and_save_split covers the bundles' records in it
@pytest.mark.pipeline
def test_generation_script_takes_the_dataset_key_from_the_selected_config(
    tmp_path: Path,
) -> None:
    save_path = tmp_path / "datasets" / "cube_without_noise"
    command = [
        sys.executable,
        "scripts/dataset_generation/generate_dataset.py",
        "dataset=cube_without_noise",
        "dataset.kwargs.n_samples=30",
        "dataset_realization_indices=[0,1]",
        "seeds=[7,8]",
        f"save_path={save_path}",
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

    for index, seed in ((0, 7), (1, 8)):
        record = bundle_provenance(save_path / str(index) / "val.bundle")
        assert record is not None
        assert record.dataset_key == "cube_without_noise"
        assert record.dataset_realization_index == index
        assert record.split == "val"
        assert record.seed == seed
        assert record.resolved_config["dataset"]["kwargs"]["n_samples"] == 30
        assert record.resolved_config["seeds"] == [7, 8]
