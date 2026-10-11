"""
Dataset generation fixes feature costs in the dataset bundles (ADR 0011).

That a bundle saved without feature costs does not load is tested once
against the in-memory base in `test_in_memory_dataset.py`.
"""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.registry import get_class
from afabench.datasets.feature_costs import FeatureCostsConflictError
from test.src.common.datasets.test_generation_indices import (
    DATASET_CLASS_NAMES,
    SPLITS,
    dataset_kwargs,
    generate_splits,
    write_image_folders,
    write_tabular_source,
)

if TYPE_CHECKING:
    from afabench.core.types import AFADataset


def write_feature_costs_file(path: Path, costs: torch.Tensor) -> Path:
    path.write_text(",".join(str(cost) for cost in costs.flatten().tolist()))
    return path


@pytest.mark.parametrize("class_name", DATASET_CLASS_NAMES)
def test_without_a_feature_costs_file_bundles_store_the_datasets_own_costs(
    class_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = dataset_kwargs(class_name, tmp_path, monkeypatch)
    dataset_class = cast("type[AFADataset]", get_class(class_name))
    # Computed from config for CUBE-NM and CUBE-NUC, unit costs otherwise
    expected = dataset_class(**kwargs).get_feature_acquisition_costs()

    splits = generate_splits(class_name, kwargs, tmp_path / "out")

    for split in SPLITS:
        torch.testing.assert_close(
            splits[split].get_feature_acquisition_costs(), expected
        )


def test_bundles_keep_the_feature_costs_file_after_it_is_gone(
    tmp_path: Path,
) -> None:
    kwargs = write_tabular_source("CKDDataset", tmp_path / "source.csv")
    costs = torch.tensor([1.0, 2.0, 0.5, 4.0, 0.0, 3.0])
    costs_file = write_feature_costs_file(
        tmp_path / "feature_costs.csv", costs
    )

    generate_splits("CKDDataset", kwargs, tmp_path / "out", costs_file)
    costs_file.unlink()

    for split in SPLITS:
        loaded = cast(
            "AFADataset",
            load_bundle(tmp_path / "out" / f"{split}.bundle").obj,
        )
        torch.testing.assert_close(
            loaded.get_feature_acquisition_costs(), costs
        )


def test_image_dataset_bundles_store_the_feature_costs_file(
    tmp_path: Path,
) -> None:
    kwargs = write_image_folders(tmp_path / "images")
    costs = torch.rand(
        (3, 224, 224), generator=torch.Generator().manual_seed(0)
    )
    costs_file = write_feature_costs_file(
        tmp_path / "feature_costs.csv", costs
    )

    splits = generate_splits(
        "ImagenetteDataset", kwargs, tmp_path / "out", costs_file
    )

    for split in SPLITS:
        torch.testing.assert_close(
            splits[split].get_feature_acquisition_costs(), costs
        )


def test_a_feature_costs_file_for_a_dataset_that_computes_its_own_raises(
    tmp_path: Path,
) -> None:
    costs_file = write_feature_costs_file(
        tmp_path / "feature_costs.csv", torch.ones(55)
    )

    with pytest.raises(FeatureCostsConflictError, match=r"feature_costs\.csv"):
        generate_splits(
            "CubeNMDataset",
            {"n_samples": 20, "seed": 3},
            tmp_path / "out",
            costs_file,
        )


@pytest.mark.parametrize(
    ("costs", "message"),
    [
        (torch.ones(5), "has 5 values, expected 6"),
        (torch.tensor([1.0, 1.0, -1.0, 1.0, 1.0, 1.0]), "non-negative"),
    ],
)
def test_an_invalid_feature_costs_file_names_it(
    costs: torch.Tensor, message: str, tmp_path: Path
) -> None:
    kwargs = write_tabular_source("CKDDataset", tmp_path / "source.csv")
    costs_file = write_feature_costs_file(tmp_path / "bad_costs.csv", costs)

    with pytest.raises(ValueError, match=message) as error:
        generate_splits("CKDDataset", kwargs, tmp_path / "out", costs_file)
    assert "bad_costs.csv" in str(error.value)
