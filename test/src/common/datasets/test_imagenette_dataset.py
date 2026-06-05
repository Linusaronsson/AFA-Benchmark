from pathlib import Path
from typing import cast

import torch

from afabench.datasets.datasets import ImagenetteDataset


def make_imagenette_dataset(indices: list[int]) -> ImagenetteDataset:
    dataset = ImagenetteDataset.__new__(ImagenetteDataset)
    dataset.data_root = "extra/data/"
    dataset.variant_dir = "imagenette2-320"
    dataset.load_subdirs = ("train",)
    dataset.image_size = 224
    dataset.split_role = "train"
    dataset.transform = None
    dataset.samples = [Path(f"image_{i}.JPEG") for i in range(5)]
    dataset.targets = torch.arange(5, dtype=torch.long)
    dataset.indices = torch.tensor(indices, dtype=torch.long)
    return cast("ImagenetteDataset", dataset)


def test_imagenette_create_subset_reduces_visible_indices() -> None:
    dataset = make_imagenette_dataset([4, 2, 0, 1])

    subset = dataset.create_subset([0, 2])

    assert subset.samples == dataset.samples
    assert torch.equal(subset.targets, dataset.targets)
    assert torch.equal(subset.indices, torch.tensor([4, 0]))
    assert len(subset) == 2
