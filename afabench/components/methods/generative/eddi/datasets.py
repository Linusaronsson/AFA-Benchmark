from typing import Any

from torch.utils.data import DataLoader

from afabench.core.types import AFADataset
from afabench.training.tensor_batches import (
    TensorBatchDataset,
    passthrough_batch,
)


def prepare_datasets(
    train_dataset: AFADataset, val_dataset: AFADataset, batch_size: int
) -> tuple[
    DataLoader[Any],
    DataLoader[Any],
    int,
    int,
]:
    # Get dimensions using shape properties
    d_in = train_dataset.feature_shape.numel()
    d_out = train_dataset.label_shape[0]

    def converted_dataset(dataset: AFADataset) -> TensorBatchDataset:
        features, labels = dataset.get_all_data()
        return TensorBatchDataset(
            features.float(), labels.argmax(dim=1).long()
        )

    train_loader = DataLoader(
        converted_dataset(train_dataset),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=passthrough_batch,
    )
    val_loader = DataLoader(
        converted_dataset(val_dataset),
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=passthrough_batch,
    )

    return train_loader, val_loader, d_in, d_out
