"""The in-memory base behind the AFADataset classes (ADR 0004, ADR 0011)."""

from pathlib import Path
from typing import ClassVar, final, override

import pytest
import torch
from torch import Tensor

from afabench.datasets.feature_costs import MissingFeatureCostsError
from afabench.datasets.in_memory import (
    InMemoryAFADataset,
    MissingConfigKeyError,
)
from afabench.datasets.utils import MissingGenerationIndicesError

N_INSTANCES = 6


class TinyDataset(InMemoryAFADataset):
    """Instance i has features (i, i + 0.5, i + offset) and label i % 2."""

    config_keys: ClassVar[tuple[str, ...]] = ("n_samples", "offset")

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return False

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([3])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([2])

    def __init__(self, n_samples: int = N_INSTANCES, offset: float = 0.0):
        self.n_samples = n_samples
        self.offset = offset
        super().__init__()

    @override
    def _build(self) -> tuple[Tensor, Tensor]:
        instance = torch.arange(self.n_samples, dtype=torch.float32)
        features = torch.stack(
            [instance, instance + 0.5, instance + self.offset], dim=1
        )
        labels = torch.nn.functional.one_hot(
            torch.arange(self.n_samples) % 2, num_classes=2
        ).float()
        return features, labels


def test_a_constructed_dataset_numbers_its_instances_in_generation_order() -> (
    None
):
    dataset = TinyDataset()

    assert len(dataset) == N_INSTANCES
    assert dataset.get_generation_indices().tolist() == list(range(6))
    features, label = dataset[4]
    assert features.tolist() == [4.0, 4.5, 4.0]
    assert label.tolist() == [1.0, 0.0]
    all_features, all_labels = dataset.get_all_data()
    assert all_features.shape == (N_INSTANCES, 3)
    assert all_labels.shape == (N_INSTANCES, 2)


def test_create_subset_composes_generation_indices_and_selects_the_data() -> (
    None
):
    dataset = TinyDataset()

    subset = dataset.create_subset([5, 2, 4, 1]).create_subset([3, 0])

    assert subset.get_generation_indices().tolist() == [1, 5]
    assert len(subset) == 2
    features, labels = subset.get_all_data()
    assert features[:, 0].tolist() == [1.0, 5.0]
    assert labels.argmax(-1).tolist() == [1, 1]


def test_create_subset_leaves_the_config_untouched() -> None:
    dataset = TinyDataset(n_samples=N_INSTANCES, offset=1.5)

    subset = dataset.create_subset([0, 1])

    assert subset.config == {"n_samples": N_INSTANCES, "offset": 1.5}
    assert dataset.config == subset.config


def test_a_build_with_the_wrong_feature_shape_names_the_dataset() -> None:
    @final
    class WrongShapeDataset(TinyDataset):
        @property
        @override
        def feature_shape(self) -> torch.Size:
            return torch.Size([4])

    with pytest.raises(ValueError, match=r"WrongShapeDataset.*\(3,\).*\(4,\)"):
        WrongShapeDataset()


def save_subset(tmp_path: Path) -> tuple[TinyDataset, Path]:
    """Save a shuffled subset with non-unit feature costs, as dataset generation does."""
    subset = TinyDataset(offset=1.5).create_subset([5, 1, 3])
    subset.feature_costs = torch.tensor([1.0, 2.0, 0.5])
    subset.save(tmp_path)
    return subset, tmp_path / "dataset.pt"


def test_save_and_load_round_trip_the_dataset(tmp_path: Path) -> None:
    subset, _ = save_subset(tmp_path)

    loaded = TinyDataset.load(tmp_path)

    features, labels = subset.get_all_data()
    loaded_features, loaded_labels = loaded.get_all_data()
    torch.testing.assert_close(loaded_features, features)
    torch.testing.assert_close(loaded_labels, labels)
    assert loaded.get_generation_indices().tolist() == [5, 1, 3]
    torch.testing.assert_close(
        loaded.get_feature_acquisition_costs(), torch.tensor([1.0, 2.0, 0.5])
    )
    assert loaded.config == {"n_samples": N_INSTANCES, "offset": 1.5}
    assert loaded.offset == 1.5


def test_dataset_pt_holds_the_tensors_the_feature_costs_and_the_config(
    tmp_path: Path,
) -> None:
    _, file = save_subset(tmp_path)

    data = torch.load(file)

    assert set(data) == {
        "features",
        "labels",
        "generation_indices",
        "feature_costs",
        "config",
    }
    assert data["config"] == {"n_samples": N_INSTANCES, "offset": 1.5}


@pytest.mark.parametrize(
    ("missing", "error"),
    [
        ("generation_indices", MissingGenerationIndicesError),
        ("feature_costs", MissingFeatureCostsError),
    ],
)
def test_loading_a_bundle_saved_without_a_required_entry_names_the_file(
    missing: str, error: type[Exception], tmp_path: Path
) -> None:
    _, file = save_subset(tmp_path)
    data = torch.load(file)
    del data[missing]
    torch.save(data, file)

    with pytest.raises(error, match=r"dataset\.pt"):
        TinyDataset.load(tmp_path)


def test_loading_a_config_without_a_declared_key_names_the_key(
    tmp_path: Path,
) -> None:
    _, file = save_subset(tmp_path)
    data = torch.load(file)
    del data["config"]["offset"]
    torch.save(data, file)

    with pytest.raises(MissingConfigKeyError, match=r"'offset'") as error:
        TinyDataset.load(tmp_path)
    assert "dataset.pt" in str(error.value)
