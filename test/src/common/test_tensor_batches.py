import torch
from torch.utils.data import DataLoader, TensorDataset

from afabench.training.tensor_batches import (
    TensorBatchDataset,
    passthrough_batch,
)


def test_vectorized_batches_equal_default_collation() -> None:
    features = torch.arange(70, dtype=torch.float32).reshape(10, 7)
    labels = torch.arange(10)
    legacy_generator = torch.Generator().manual_seed(4)
    vectorized_generator = torch.Generator().manual_seed(4)
    legacy = DataLoader(
        TensorDataset(features, labels),
        batch_size=3,
        shuffle=True,
        drop_last=True,
        generator=legacy_generator,
    )
    vectorized = DataLoader(
        TensorBatchDataset(features, labels),
        batch_size=3,
        shuffle=True,
        drop_last=True,
        generator=vectorized_generator,
        collate_fn=passthrough_batch,
    )

    for legacy_batch, vectorized_batch in zip(legacy, vectorized, strict=True):
        assert all(
            torch.equal(left, right)
            for left, right in zip(legacy_batch, vectorized_batch, strict=True)
        )


def test_vectorized_batches_preserve_training_trajectory() -> None:
    features = torch.arange(70, dtype=torch.float32).reshape(10, 7) / 70
    labels = torch.arange(10) % 2
    initial = torch.nn.Linear(7, 2)
    legacy_model = torch.nn.Linear(7, 2)
    vectorized_model = torch.nn.Linear(7, 2)
    legacy_model.load_state_dict(initial.state_dict())
    vectorized_model.load_state_dict(initial.state_dict())
    legacy_optimizer = torch.optim.SGD(legacy_model.parameters(), lr=0.1)
    vectorized_optimizer = torch.optim.SGD(
        vectorized_model.parameters(), lr=0.1
    )
    legacy = DataLoader(
        TensorDataset(features, labels),
        batch_size=3,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(4),
    )
    vectorized = DataLoader(
        TensorBatchDataset(features, labels),
        batch_size=3,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(4),
        collate_fn=passthrough_batch,
    )

    for model, optimizer, loader in (
        (legacy_model, legacy_optimizer, legacy),
        (vectorized_model, vectorized_optimizer, vectorized),
    ):
        for batch_features, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(
                model(batch_features), batch_labels
            ).backward()
            optimizer.step()

    for legacy_parameter, vectorized_parameter in zip(
        legacy_model.parameters(), vectorized_model.parameters(), strict=True
    ):
        assert torch.equal(legacy_parameter, vectorized_parameter)
