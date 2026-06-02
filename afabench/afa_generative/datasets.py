from typing import final, override

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from afabench.common.custom_types import AFADataset


def prepare_datasets(
    train_dataset: AFADataset, val_dataset: AFADataset, batch_size: int
) -> tuple[
    DataLoader[tuple[Tensor, Tensor]],
    DataLoader[tuple[Tensor, Tensor]],
    int,
    int,
]:
    # Get dimensions using shape properties
    d_in = train_dataset.feature_shape[0]
    d_out = train_dataset.label_shape[0]

    # Create new datasets with converted data format
    @final
    class ConvertedDataset(torch.utils.data.Dataset[tuple[Tensor, Tensor]]):
        def __init__(self, original_dataset: AFADataset):
            self.original_dataset: AFADataset = original_dataset
            self.features, self.labels = original_dataset.get_all_data()
            self.features = self.features.float()
            self.labels = self.labels.argmax(dim=1).long()

        @override
        def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
            return self.features[idx], self.labels[idx]

        def __len__(self) -> int:
            return len(self.original_dataset)

    converted_train_dataset = ConvertedDataset(train_dataset)
    converted_val_dataset = ConvertedDataset(val_dataset)

    train_loader = DataLoader(
        converted_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        converted_val_dataset, batch_size=batch_size, pin_memory=True
    )

    return train_loader, val_loader, d_in, d_out
