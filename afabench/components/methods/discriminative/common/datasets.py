from typing import Any

from torch.utils.data import DataLoader

from afabench.core.types import AFADataset
from afabench.training.smoke_test import dataset_subset, training_batch_size


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

    # Create new datasets with converted data format
    class ConvertedDataset:
        source_availability: Any
        selection_availability: Any

        def __init__(self, original_dataset: AFADataset):
            self.original_dataset: Any = original_dataset
            self.features, self.labels = original_dataset.get_all_data()
            self.features: Any = self.features.float()
            self.labels: Any = self.labels.argmax(dim=1).long()
            self.source_availability = getattr(
                original_dataset,
                "source_availability",
                None,
            )
            self.selection_availability = getattr(
                original_dataset,
                "selection_availability",
                None,
            )

        def __getitem__(self, idx: int):
            if self.source_availability is not None:
                assert self.selection_availability is not None
                return (
                    self.features[idx],
                    self.labels[idx],
                    self.source_availability[idx],
                    self.selection_availability[idx],
                )
            return self.features[idx], self.labels[idx]

        def __len__(self):
            return len(self.original_dataset)

    train_dataset = ConvertedDataset(train_dataset)
    val_dataset = ConvertedDataset(val_dataset)

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
