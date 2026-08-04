from typing import Any

from torch.utils.data import DataLoader

from afabench.core.types import AFADataset
from afabench.training.smoke_test import dataset_subset, training_batch_size
from afabench.training.tensor_batches import (
    TensorBatchDataset,
    passthrough_batch,
)


def prepare_datasets(
    train_dataset,  # noqa: ANN001
    val_dataset,  # noqa: ANN001
    batch_size: int,
    *,
    smoke_test: bool = False,
) -> tuple[DataLoader[Any], DataLoader[Any], int, int]:
    # Get dimensions using shape properties
    d_in = train_dataset.feature_shape.numel()
    d_out = train_dataset.label_shape[0]
    train_dataset = dataset_subset(train_dataset, smoke_test=smoke_test)
    val_dataset = dataset_subset(val_dataset, smoke_test=smoke_test)
    batch_size = training_batch_size(
        smoke_test=smoke_test,
        default_batch_size=batch_size,
    )

    def converted_dataset(dataset: AFADataset) -> TensorBatchDataset:
        features, labels = dataset.get_all_data()
        source_availability = getattr(dataset, "source_availability", None)
        selection_availability = getattr(
            dataset, "selection_availability", None
        )
        tensors = [features.float(), labels.argmax(dim=1).long()]
        if source_availability is not None:
            assert selection_availability is not None
            tensors.extend([source_availability, selection_availability])
        return TensorBatchDataset(*tensors)

    train_dataset = converted_dataset(train_dataset)
    val_dataset = converted_dataset(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=passthrough_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=passthrough_batch,
    )

    return train_loader, val_loader, d_in, d_out
