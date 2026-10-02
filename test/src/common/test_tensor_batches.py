from collections.abc import Callable, Iterable
from typing import final, override

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from afabench.components.methods.discriminative.common.datasets import (
    prepare_datasets as prepare_discriminative_datasets,
)
from afabench.components.methods.generative.eddi.datasets import (
    prepare_datasets as prepare_eddi_datasets,
)
from afabench.core.types import AFADataset
from afabench.datasets.datasets import CubeDataset
from afabench.training.tensor_batches import (
    TensorBatchDataset,
    passthrough_batch,
)

type PrepareDatasets = Callable[
    [AFADataset, AFADataset, int],
    tuple[DataLoader[object], DataLoader[object], int, int],
]


@final
class _PerInstanceDataset(Dataset[tuple[Tensor, Tensor]]):
    """Reference path: index one instance at a time, then default-collate."""

    def __init__(self, dataset: AFADataset) -> None:
        self.dataset = dataset

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        features, label = self.dataset[index]
        return features.float(), label.argmax().long()

    def __len__(self) -> int:
        return len(self.dataset)


def _seeded_batches(
    loader: Iterable[object], seed: int
) -> list[tuple[Tensor, ...]]:
    torch.manual_seed(seed)
    return [tuple(batch) for batch in loader]  # pyright: ignore[reportArgumentType]


def _assert_same_batches(
    actual: list[tuple[Tensor, ...]], expected: list[tuple[Tensor, ...]]
) -> None:
    assert len(actual) == len(expected)
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        assert len(actual_batch) == len(expected_batch)
        for actual_tensor, expected_tensor in zip(
            actual_batch, expected_batch, strict=True
        ):
            assert actual_tensor.dtype == expected_tensor.dtype
            assert torch.equal(actual_tensor, expected_tensor)


def test_tensor_batch_dataset_matches_per_instance_collation() -> None:
    features = torch.arange(70, dtype=torch.float32).reshape(10, 7)
    labels = torch.arange(10)
    per_instance = DataLoader(
        TensorDataset(features, labels),
        batch_size=3,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(4),
    )
    batched = DataLoader(
        TensorBatchDataset(features, labels),
        batch_size=3,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(4),
        collate_fn=passthrough_batch,
    )

    _assert_same_batches(
        [tuple(batch) for batch in batched],
        [tuple(batch) for batch in per_instance],
    )


def test_tensor_batch_dataset_rejects_misaligned_tensors() -> None:
    with pytest.raises(ValueError, match="leading dimension"):
        TensorBatchDataset(torch.zeros(3, 2), torch.zeros(4))


@pytest.mark.parametrize(
    "prepare_datasets",
    [prepare_discriminative_datasets, prepare_eddi_datasets],
    ids=["discriminative", "eddi"],
)
def test_prepared_loaders_match_per_instance_loaders(
    prepare_datasets: PrepareDatasets,
) -> None:
    train_dataset = CubeDataset(n_samples=10, seed=42)
    val_dataset = CubeDataset(n_samples=7, seed=43)
    batch_size = 3

    train_loader, val_loader, _, _ = prepare_datasets(
        train_dataset, val_dataset, batch_size
    )
    per_instance_train_loader = DataLoader(
        _PerInstanceDataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    per_instance_val_loader = DataLoader(
        _PerInstanceDataset(val_dataset), batch_size=batch_size
    )

    train_batches = _seeded_batches(train_loader, seed=0)
    _assert_same_batches(
        train_batches, _seeded_batches(per_instance_train_loader, seed=0)
    )
    # drop_last discards the incomplete final training batch.
    assert len(train_batches) == 3
    # A different seed reshuffles the training batches.
    assert not torch.equal(
        torch.cat([batch[0] for batch in train_batches]),
        torch.cat(
            [batch[0] for batch in _seeded_batches(train_loader, seed=1)]
        ),
    )

    val_batches = _seeded_batches(val_loader, seed=0)
    _assert_same_batches(
        val_batches, _seeded_batches(per_instance_val_loader, seed=0)
    )
    # Validation keeps the incomplete final batch.
    assert [len(batch[0]) for batch in val_batches] == [3, 3, 1]
