import importlib
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
from afabench.components.methods.rl.common.dataset_utils import (
    DataModuleFromDatasets,
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


@pytest.mark.parametrize("num_workers", [0, 2])
def test_tensor_batch_dataset_matches_per_instance_collation(
    num_workers: int,
) -> None:
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
        num_workers=num_workers,
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


@pytest.mark.parametrize("persistent_workers", [False, True])
def test_classifier_datamodule_multiple_workers_across_epochs(
    persistent_workers: bool,
) -> None:
    """The classifier-training data module works with num_workers > 1."""
    n_train, n_val, n_features = 20, 13, 7
    train_features = torch.arange(
        n_train * n_features, dtype=torch.float32
    ).reshape(n_train, n_features)
    train_labels = torch.arange(n_train)
    val_features = torch.arange(
        n_val * n_features, dtype=torch.float32
    ).reshape(n_val, n_features)
    val_labels = torch.arange(n_val)

    datamodule = DataModuleFromDatasets(
        train_dataset=TensorBatchDataset(train_features, train_labels),
        val_dataset=TensorBatchDataset(val_features, val_labels),
        batch_size=4,
        num_workers=2,
        persistent_workers=persistent_workers,
        collate_fn=passthrough_batch,
    )
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    for _ in range(3):
        seen_train_labels = []
        for batch_features, batch_labels in train_loader:
            # Feature/label pairing survives worker dispatch.
            assert torch.equal(batch_features, train_features[batch_labels])
            seen_train_labels.append(batch_labels)
        # Every training instance is seen exactly once per epoch.
        assert torch.equal(
            torch.sort(torch.cat(seen_train_labels)).values, train_labels
        )

        seen_val_labels = []
        for batch_features, batch_labels in val_loader:
            assert torch.equal(batch_features, val_features[batch_labels])
            seen_val_labels.append(batch_labels)
        # Every validation instance is seen exactly once per epoch.
        assert torch.equal(
            torch.sort(torch.cat(seen_val_labels)).values, val_labels
        )


def test_classifier_datamodule_workers_fork_after_torchrl_import() -> None:
    """Workers fork even though importing torchrl makes spawn the default."""
    _ = importlib.import_module("torchrl")

    datamodule = DataModuleFromDatasets(
        train_dataset=TensorBatchDataset(torch.zeros(4, 2), torch.zeros(4)),
        val_dataset=TensorBatchDataset(torch.zeros(4, 2), torch.zeros(4)),
        num_workers=2,
        collate_fn=passthrough_batch,
    )

    for loader in [datamodule.train_dataloader(), datamodule.val_dataloader()]:
        context = loader.multiprocessing_context
        assert context is not None
        assert context.get_start_method() == "fork"


def test_classifier_datamodule_rejects_persistent_workers_without_workers() -> (
    None
):
    """persistent_workers=True with num_workers=0 must fail clearly."""
    datamodule = DataModuleFromDatasets(
        train_dataset=TensorBatchDataset(torch.zeros(4, 2), torch.zeros(4)),
        val_dataset=TensorBatchDataset(torch.zeros(4, 2), torch.zeros(4)),
        num_workers=0,
        persistent_workers=True,
        collate_fn=passthrough_batch,
    )

    with pytest.raises(ValueError, match="persistent_workers"):
        datamodule.train_dataloader()
