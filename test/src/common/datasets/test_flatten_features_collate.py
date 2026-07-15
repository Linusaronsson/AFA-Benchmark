import torch

from afabench.datasets.utils import flatten_features_collate


def test_flatten_features_collate_flattens_row_masks_too() -> None:
    collate = flatten_features_collate(n_feature_dims=2)
    batch = [
        (
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([1.0, 0.0]),
            torch.tensor([[True, False], [True, True]]),
        ),
        (
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([[False, True], [False, False]]),
        ),
    ]

    features, labels, availability = collate(batch)

    assert features.shape == (2, 4)
    assert labels.shape == (2, 2)
    assert availability.shape == (2, 4)
    assert torch.equal(
        availability,
        torch.tensor([[True, False, True, True], [False, True, False, False]]),
    )
