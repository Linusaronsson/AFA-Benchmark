"""Every instance carries its generation index through dataset bundles (ADR 0004)."""

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import datasets

from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.core.registry import REGISTERED_CLASSES, get_class
from afabench.core.types import AFADataset
from afabench.datasets.config import SplitRatioConfig
from afabench.datasets.datasets import CubeDataset, ImagenetteDataset
from afabench.datasets.utils import MissingGenerationIndicesError
from scripts.dataset_generation.generate_dataset import generate_and_save_split
from scripts.dataset_generation.generate_image_dataset import (
    generate_and_save_image_split,
)

N_ROWS = 20
SPLITS = ("train", "val", "test")
SPLIT_RATIO = SplitRatioConfig(train=0.6, val=0.2, test=0.2)

# (number of features, labels per row) of each real-world tabular source
TABULAR_SOURCES: dict[str, tuple[int, list[object]]] = {
    "DiabetesDataset": (45, [i % 3 for i in range(N_ROWS)]),
    "MiniBooNEDataset": (50, [i % 2 for i in range(N_ROWS)]),
    "PhysionetDataset": (41, [i % 2 for i in range(N_ROWS)]),
    "BankMarketingDataset": (
        5,
        ["yes" if i % 3 == 0 else "no" for i in range(N_ROWS)],
    ),
    "CKDDataset": (6, [i % 2 for i in range(N_ROWS)]),
    "ACTG175Dataset": (7, [(i // 2) % 2 for i in range(N_ROWS)]),
}


def write_tabular_source(class_name: str, path: Path) -> dict[str, Any]:
    """Write a CSV source whose first feature is the row number."""
    n_features, labels = TABULAR_SOURCES[class_name]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        rng.normal(size=(N_ROWS, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    frame["f0"] = np.arange(N_ROWS, dtype=float)
    if class_name == "BankMarketingDataset":
        frame["y"] = labels
        frame.to_csv(path, sep=";", index=False)
        return {"path": str(path)}
    frame["target"] = labels
    frame.to_csv(path, index=False)
    if class_name in {"BankMarketingDataset", "CKDDataset", "ACTG175Dataset"}:
        return {"path": str(path)}
    return {"root": str(path)}


def fake_torchvision_mnist(
    root: str,  # noqa: ARG001
    train: bool,  # noqa: ARG001
    transform: object,  # noqa: ARG001
    download: bool,  # noqa: ARG001
) -> list[tuple[torch.Tensor, int]]:
    return [
        (torch.full((1, 28, 28), i / N_ROWS), i % 10) for i in range(N_ROWS)
    ]


def write_image_folders(root: Path) -> dict[str, Any]:
    """Write a tiny Imagenette-like tree: 6 images in train/, 4 in val/."""
    for subdir, n_per_class in (("train", 3), ("val", 2)):
        for class_index in range(2):
            directory = root / "variant" / subdir / f"class_{class_index}"
            directory.mkdir(parents=True)
            for i in range(n_per_class):
                Image.new("RGB", (8, 8), color=(i, class_index, 0)).save(
                    directory / f"{subdir}_{i}.png"
                )
    return {"data_root": str(root), "variant_dir": "variant"}


def dataset_kwargs(
    class_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    if class_name in TABULAR_SOURCES:
        return write_tabular_source(class_name, tmp_path / "source.csv")
    if class_name in {"MNISTDataset", "FashionMNISTDataset"}:
        torchvision_name = class_name.removesuffix("Dataset")
        monkeypatch.setattr(datasets, torchvision_name, fake_torchvision_mnist)
        return {"root": str(tmp_path)}
    if class_name == "ImagenetteDataset":
        return write_image_folders(tmp_path / "images")
    # Synthetic datasets
    return {"n_samples": N_ROWS, "seed": 3}


def generate_splits(
    class_name: str, kwargs: dict[str, Any], save_path: Path
) -> dict[str, AFADataset]:
    dataset_class = cast("type[AFADataset]", get_class(class_name))
    generate = (
        generate_and_save_image_split
        if class_name == "ImagenetteDataset"
        else generate_and_save_split
    )
    generate(
        dataset_class=dataset_class,
        split_ratio=SPLIT_RATIO,
        seed_for_split=5,
        save_path=save_path,
        dataset_kwargs=kwargs,
        metadata_to_save={},
    )
    return {
        split: cast(
            "AFADataset", load_bundle(save_path / f"{split}.bundle")[0]
        )
        for split in SPLITS
    }


# Every registered dataset class, once even if it has aliases
DATASET_CLASS_NAMES = sorted(
    {
        path.rsplit(".", 1)[1]
        for path in REGISTERED_CLASSES.values()
        if path.startswith("afabench.datasets.")
    }
)


@pytest.mark.parametrize("class_name", DATASET_CLASS_NAMES)
def test_splits_partition_the_generated_dataset(
    class_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = dataset_kwargs(class_name, tmp_path, monkeypatch)

    splits = generate_splits(class_name, kwargs, tmp_path / "out")

    per_split = [splits[s].get_generation_indices().tolist() for s in SPLITS]
    for split, indices in zip(SPLITS, per_split, strict=True):
        assert len(indices) == len(splits[split]), split
    all_indices = [i for indices in per_split for i in indices]
    # 6 train/ plus 4 val/ images for Imagenette, N_ROWS otherwise
    n_generated = 10 if class_name == "ImagenetteDataset" else N_ROWS
    assert sorted(all_indices) == list(range(n_generated))


@pytest.mark.parametrize("class_name", DATASET_CLASS_NAMES)
def test_generation_indices_locate_instances_in_the_generated_dataset(
    class_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = dataset_kwargs(class_name, tmp_path, monkeypatch)
    dataset_class = cast("type[AFADataset]", get_class(class_name))

    splits = generate_splits(class_name, kwargs, tmp_path / "out")

    if class_name == "ImagenetteDataset":
        # Images are augmented on access, so compare the files instead
        test = cast("ImagenetteDataset", splits["test"])
        assert all(
            "/val/" in str(test.samples[i]) for i in test.generation_indices
        )
        return
    source_features, source_labels = dataset_class(**kwargs).get_all_data()
    for split in SPLITS:
        features, labels = splits[split].get_all_data()
        generation_indices = splits[split].get_generation_indices()
        torch.testing.assert_close(
            features, source_features[generation_indices]
        )
        torch.testing.assert_close(labels, source_labels[generation_indices])


@pytest.mark.parametrize("class_name", sorted(TABULAR_SOURCES))
def test_real_world_tabular_datasets_keep_source_order(
    class_name: str, tmp_path: Path
) -> None:
    kwargs = write_tabular_source(class_name, tmp_path / "source.csv")
    _, labels = TABULAR_SOURCES[class_name]

    dataset = cast("type[AFADataset]", get_class(class_name))(**kwargs)

    features, one_hot_labels = dataset.get_all_data()
    # Feature f0 is the source row number; normalization keeps its order
    assert len(dataset) == N_ROWS
    assert bool((features[1:, 0] > features[:-1, 0]).all())
    expected = [{"yes": 1, "no": 0}.get(str(label), label) for label in labels]
    assert one_hot_labels.argmax(-1).tolist() == expected


def imagenette(tmp_path: Path) -> ImagenetteDataset:
    return ImagenetteDataset(
        **write_image_folders(tmp_path / "images"), split_role="val"
    )


def cube(tmp_path: Path) -> CubeDataset:  # noqa: ARG001
    return CubeDataset(n_samples=N_ROWS, seed=1)


@pytest.mark.parametrize("make_dataset", [cube, imagenette])
def test_create_subset_composes_generation_indices(
    make_dataset: Callable[[Path], AFADataset], tmp_path: Path
) -> None:
    dataset = make_dataset(tmp_path)

    subset = dataset.create_subset([7, 2, 9, 4]).create_subset([3, 0])

    assert subset.get_generation_indices().tolist() == [4, 7]


@pytest.mark.parametrize("make_dataset", [cube, imagenette])
def test_loading_a_bundle_without_generation_indices_names_it(
    make_dataset: Callable[[Path], AFADataset], tmp_path: Path
) -> None:
    bundle_path = tmp_path / "old_split.bundle"
    save_bundle(make_dataset(tmp_path), bundle_path, metadata={})
    data_path = bundle_path / "data" / "dataset.pt"
    data = torch.load(data_path)
    del data["generation_indices"]
    torch.save(data, data_path)

    with pytest.raises(
        MissingGenerationIndicesError, match=r"old_split\.bundle"
    ):
        load_bundle(bundle_path)


def test_bundle_round_trip_keeps_generation_indices(tmp_path: Path) -> None:
    subset = CubeDataset(n_samples=N_ROWS, seed=1).create_subset([5, 1, 3])
    bundle_path = tmp_path / "split.bundle"
    save_bundle(subset, bundle_path, metadata={})

    loaded = cast("AFADataset", load_bundle(bundle_path)[0])

    assert loaded.get_generation_indices().tolist() == [5, 1, 3]
