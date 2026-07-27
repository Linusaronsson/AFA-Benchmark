import hashlib
from pathlib import Path

import pytest
import torch

from afabench.core.bundle_system.bundle import load_bundle
from afabench.datasets.config import SplitRatioConfig
from afabench.datasets.datasets import NHANESMortalityDataset
from scripts.dataset_generation.generate_dataset import (
    generate_and_save_split,
)

ROOT = Path("extra/data/nhanes_mortality")
SOURCE = ROOT / "source"
EXPECTED_HASHES = {
    "X_nhanes_binary.csv": (
        "22074d137dadfdf18a949c84f7d5efc6ce68ca731472f2b1cbc7c9d3271af4b4"
    ),
    "feature_costs.txt": (
        "0b47963b6ad43dc98b7d0fb2065af47bba8727cb34bee5da27e148ef2abd6be6"
    ),
    "feature_groups.txt": (
        "9f10700dd06bc432b0fa382754d7a4c5647a8cfe8f0d535b87e5b4f370f574ad"
    ),
    "y_nhanes_binary.npy": (
        "5f6ad190e7a97d37ce1de17a54f912a823b72b60391d2fbdfcfa5491a1561d5e"
    ),
}


def dataset_kwargs() -> dict[str, str]:
    return {
        "features_path": str(SOURCE / "X_nhanes_binary.csv"),
        "labels_path": str(SOURCE / "y_nhanes_binary.npy"),
        "schema_path": str(ROOT / "schema.csv"),
    }


def test_source_snapshot_checksums() -> None:
    for name, expected in EXPECTED_HASHES.items():
        digest = hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()
        assert digest == expected


def test_schema_and_source_shape() -> None:
    dataset = NHANESMortalityDataset(**dataset_kwargs())

    assert len(dataset) == 13_442
    assert dataset.feature_shape == torch.Size([118])
    assert torch.equal(dataset.labels.sum(dim=0), torch.tensor([3371, 10071]))
    assert dataset.get_missingness_group_ids().unique().numel() == 27
    selection_costs = torch.zeros(27).scatter_add(
        0,
        dataset.get_missingness_group_ids(),
        dataset.get_feature_acquisition_costs(),
    )
    assert (selection_costs > 0).all()
    assert selection_costs[16] == pytest.approx(4.071)
    assert selection_costs[11] == pytest.approx(5.111)


def test_fixed_test_split_and_train_fitted_preprocessing(
    tmp_path: Path,
) -> None:
    split = SplitRatioConfig(train=0.6, val=0.2, test=0.2)
    for instance in (0, 1):
        generate_and_save_split(
            dataset_class=NHANESMortalityDataset,
            split_ratio=split,
            seed_for_split=instance,
            save_path=tmp_path / str(instance),
            dataset_kwargs=dataset_kwargs(),
            metadata_to_save={"instance_idx": instance},
            fixed_test_seed=100,
            stratify=True,
        )

    loaded: dict[tuple[int, str], NHANESMortalityDataset] = {}
    for instance in (0, 1):
        for name in ("train", "val", "test"):
            dataset, _ = load_bundle(
                tmp_path / str(instance) / f"{name}.bundle"
            )
            assert isinstance(dataset, NHANESMortalityDataset)
            assert torch.isfinite(dataset.features).all()
            loaded[instance, name] = dataset

        train_ids = set(loaded[instance, "train"].source_row_ids.tolist())
        val_ids = set(loaded[instance, "val"].source_row_ids.tolist())
        test_ids = set(loaded[instance, "test"].source_row_ids.tolist())
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert torch.allclose(
            loaded[instance, "train"].features.mean(dim=0),
            torch.zeros(118),
            atol=1e-5,
        )

    assert torch.equal(
        loaded[0, "test"].source_row_ids,
        loaded[1, "test"].source_row_ids,
    )
    assert not torch.equal(
        loaded[0, "train"].source_row_ids,
        loaded[1, "train"].source_row_ids,
    )
    assert not torch.equal(
        loaded[0, "train"].preprocessing_mean,
        loaded[1, "train"].preprocessing_mean,
    )
