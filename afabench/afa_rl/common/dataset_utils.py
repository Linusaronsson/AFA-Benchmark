from collections.abc import Callable
from typing import final, override

import lightning as pl
import torch
from afabench.afa_rl.custom_types import AFADatasetFn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from afabench.common.custom_types import (
    Features,
    Label,
)


def get_wrapped_batch(t: torch.Tensor, idx: int, numel: int) -> torch.Tensor:
    """Get a batch of size num_elems from a tensor t, starting at index idx, wrapping around if necessary."""
    n = len(t)
    repeated = t.repeat((numel // n) + 2, *[1] * (t.ndim - 1))
    return repeated[idx : idx + numel]


def get_afa_dataset_fn(
    features: Features,
    labels: Label,
    device: torch.device | None = None,
    *,
    shuffle: bool = True,
) -> AFADatasetFn:
    """Given features and labels, return a function that can be used to get batches of AFA data."""
    idx = 0  # keep track of where in the dataset we are
    original_feature_shape = features.shape[
        1:
    ]  # Store the original feature shape (excluding batch dim)

    def afa_dataset_fn(
        batch_size: torch.Size,
        *,
        move_on: bool = True,
    ) -> tuple[Features, Label]:
        nonlocal idx, features, labels
        local_features = get_wrapped_batch(features, idx, batch_size.numel())
        local_labels = get_wrapped_batch(labels, idx, batch_size.numel())
        if move_on:
            idx = idx + batch_size.numel()
            # Reset idx if needed, also shuffling the dataset
            if idx >= len(features):
                idx = 0
                # Shuffle the dataset
                if shuffle:
                    perm = torch.randperm(len(features))
                    features = features[perm]
                    labels = labels[perm]
        local_features = local_features.reshape(
            *batch_size, *original_feature_shape
        )
        local_labels = local_labels.reshape(
            *batch_size, local_labels.shape[-1]
        )

        # Move to specified device if provided
        if device is not None:
            local_features = local_features.to(device)
            local_labels = local_labels.to(device)

        return local_features, local_labels

    return afa_dataset_fn


@final
class DataModuleFromDatasets(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: Dataset[tuple[Features, Label]],
        val_dataset: Dataset[tuple[Features, Label]],
        batch_size: int = 32,
        num_workers: int = 0,
        *,
        persistent_workers: bool = False,
        collate_fn: Callable | None = None,  # pyright: ignore[reportMissingTypeArgument]
    ):
        # TODO: does not work with num_workers > 1
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.collate_fn = collate_fn

    @override
    def prepare_data(self) -> None:
        pass

    @override
    def setup(self, stage: str) -> None:
        pass

    @override
    def train_dataloader(self) -> DataLoader[tuple[Features, Label]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
        )

    @override
    def val_dataloader(self) -> DataLoader[tuple[Features, Label]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            collate_fn=self.collate_fn,
        )

    # def __getitem__(self, index: int):
    #     img, label = self.dataset[index]
    #     one_hot_label = F.one_hot(
    #         torch.tensor(label), num_classes=self.num_classes
    #     )
    #     return img, one_hot_label

    # def __len__(self):
    #     return len(self.dataset)
