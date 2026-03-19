from typing import Any

import torch
from torch.utils.data import DataLoader

from afabench.common.custom_types import AFADataset


def prepare_datasets(
    train_dataset,  # noqa: ANN001
    val_dataset,  # noqa: ANN001
    batch_size: int,
    train_forbidden_mask: torch.Tensor | None = None,
    val_forbidden_mask: torch.Tensor | None = None,
) -> tuple[DataLoader[Any], DataLoader[Any], int, int]:
    # Get dimensions using shape properties
    d_in = train_dataset.feature_shape[0]
    d_out = train_dataset.label_shape[0]

    # Create new datasets with converted data format
    class ConvertedDataset:
        def __init__(
            self,
            original_dataset: AFADataset,
            forbidden_mask: torch.Tensor | None = None,
        ) -> None:
            self.original_dataset: Any = original_dataset
            self.features, self.labels = original_dataset.get_all_data()
            self.features: Any = self.features.float()
            self.labels: Any = self.labels.argmax(dim=1).long()
            self.forbidden_mask = forbidden_mask

        def __getitem__(self, idx: int):
            if self.forbidden_mask is None:
                return self.features[idx], self.labels[idx]
            else:
                return (
                    self.features[idx],
                    self.labels[idx],
                    self.forbidden_mask[idx],
                )

        def __len__(self):
            return len(self.original_dataset)

    train_dataset = ConvertedDataset(train_dataset, train_forbidden_mask)
    val_dataset = ConvertedDataset(val_dataset, val_forbidden_mask)

    train_loader = DataLoader(
        train_dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        pin_memory=True,
    )

    return train_loader, val_loader, d_in, d_out
