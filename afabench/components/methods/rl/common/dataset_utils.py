from collections.abc import Callable
from typing import cast, final, override

import lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from afabench.components.methods.rl.common.custom_types import (
    AFADatasetBatch,
    AFADatasetFn,
)
from afabench.core.types import (
    FeatureMask,
    Features,
    Label,
    SelectionMask,
)


def get_wrapped_batch(t: torch.Tensor, idx: int, numel: int) -> torch.Tensor:
    """Get a batch of size num_elems from a tensor t, starting at index idx, wrapping around if necessary."""
    # Gather modulo n rather than materializing `t.repeat(...)`, which copied
    # the whole dataset (twice over) just to slice `numel` rows out of it.
    # Identical output: row j of the old repeated tensor is t[j % n], so the
    # slice [idx, idx+numel) is exactly t[(idx + i) % n].
    n = len(t)
    return t[(idx + torch.arange(numel, device=t.device)) % n]


def get_afa_dataset_fn(
    features: Features,
    labels: Label,
    device: torch.device | None = None,
    *,
    shuffle: bool = True,
    selection_availability: SelectionMask | None = None,
    source_availability: FeatureMask | None = None,
) -> AFADatasetFn:
    """Given features and labels, return a function that can be used to get batches of AFA data."""
    if source_availability is not None and selection_availability is None:
        msg = "source_availability requires selection_availability."
        raise ValueError(msg)
    if (
        source_availability is not None
        and source_availability.shape != features.shape
    ):
        msg = "source_availability must match features."
        raise ValueError(msg)
    idx = 0  # keep track of where in the dataset we are
    tensors = [features, labels]
    if selection_availability is not None:
        tensors.append(selection_availability)
    if source_availability is not None:
        tensors.append(source_availability)
    original_feature_shape = features.shape[
        1:
    ]  # Store the original feature shape (excluding batch dim)

    def afa_dataset_fn(
        batch_size: torch.Size,
        *,
        move_on: bool = True,
    ) -> AFADatasetBatch:
        nonlocal idx, tensors
        local = [
            get_wrapped_batch(tensor, idx, batch_size.numel())
            for tensor in tensors
        ]
        if move_on:
            idx = idx + batch_size.numel()
            # Reset idx if needed, also shuffling the dataset
            if idx >= len(features):
                idx = 0
                # Shuffle the dataset
                if shuffle:
                    perm = torch.randperm(len(features))
                    tensors = [tensor[perm] for tensor in tensors]
        local[0] = local[0].reshape(*batch_size, *original_feature_shape)
        local[1] = local[1].reshape(*batch_size, local[1].shape[-1])

        # Move to specified device if provided
        if device is not None:
            local = [tensor.to(device) for tensor in local]

        return cast("AFADatasetBatch", tuple(local))

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
