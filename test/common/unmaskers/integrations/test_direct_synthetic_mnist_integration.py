"""Integration tests for DirectUnmasker with synthetic MNIST dataset."""

import pytest
import torch

from afabench.common.datasets.datasets import SyntheticMNISTDataset
from afabench.common.unmaskers.direct_unmasker import DirectUnmasker


@pytest.fixture
def synthetic_mnist_dataset() -> SyntheticMNISTDataset:
    """Create a small synthetic MNIST dataset for testing."""
    return SyntheticMNISTDataset(seed=42, n_samples=10)


@pytest.fixture
def mnist_data(
    synthetic_mnist_dataset: SyntheticMNISTDataset,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get all features and labels from the dataset."""
    return synthetic_mnist_dataset.get_all_data()


@pytest.fixture
def single_sample(
    mnist_data: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a single sample for testing."""
    features, labels = mnist_data
    return features[0:1], labels[0:1]  # Shape: [1, 1, 28, 28], [1, 10]


@pytest.fixture
def batch_samples(
    mnist_data: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of samples for testing."""
    features, labels = mnist_data
    return features[0:3], labels[0:3]  # Shape: [3, 1, 28, 28], [3, 10]


@pytest.fixture
def direct_unmasker() -> DirectUnmasker:
    """Create a DirectUnmasker instance."""
    return DirectUnmasker()


def test_direct_unmasker_with_synthetic_mnist_single_sample(
    direct_unmasker: DirectUnmasker,
    single_sample: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test DirectUnmasker with single synthetic MNIST sample."""
    single_features, _single_labels = single_sample

    # For DirectUnmasker with image data, we need to flatten
    features_flat = single_features.view(1, 784)

    n_selections = direct_unmasker.get_n_selections(torch.Size([784]))
    assert n_selections == 784

    # Create initial feature mask (all False)
    feature_mask = torch.zeros(1, 784, dtype=torch.bool)
    masked_features = features_flat * feature_mask.float()

    # Test selection
    afa_selection = torch.tensor([[1]])  # Select feature 0 (1-indexed)
    selection_mask = torch.tensor([[True]])

    new_feature_mask = direct_unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features_flat,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=torch.Size([784]),
    )

    # Check that feature 0 is now unmasked
    assert new_feature_mask.shape == (1, 784)
    assert new_feature_mask[0, 0]
    assert torch.sum(new_feature_mask.int()) == 1


def test_direct_unmasker_with_synthetic_mnist_batch(
    direct_unmasker: DirectUnmasker,
    batch_samples: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test DirectUnmasker with batch of synthetic MNIST samples."""
    batch_features, _batch_labels = batch_samples
    batch_size = 3

    # For DirectUnmasker with image data, we need to flatten
    features_flat = batch_features.view(batch_size, 784)

    # Create initial feature mask (all False)
    feature_mask = torch.zeros(batch_size, 784, dtype=torch.bool)
    masked_features = features_flat * feature_mask.float()

    # Test different selections for each sample in batch
    afa_selection = torch.tensor([[1], [100], [500]])  # Different features
    selection_mask = torch.tensor([[True], [True], [True]])

    new_feature_mask = direct_unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features_flat,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=torch.Size([784]),
    )

    # Check output shape
    assert new_feature_mask.shape == (batch_size, 784)

    # Check that each sample has exactly one feature unmasked
    for i in range(batch_size):
        feature_sum = torch.sum(new_feature_mask[i].int())
        assert feature_sum == 1

    # Check that different features are unmasked for each sample
    assert new_feature_mask[0, 0]  # Feature 0 for first sample
    assert new_feature_mask[
        1, 99
    ]  # Feature 99 for second sample (1-indexed -> 0-indexed)
    assert new_feature_mask[2, 499]  # Feature 499 for third sample


def test_direct_unmasker_selection_ranges(
    direct_unmasker: DirectUnmasker,
) -> None:
    """Test that DirectUnmasker handles the correct selection range."""
    feature_shape = torch.Size([784])
    n_selections = direct_unmasker.get_n_selections(feature_shape)
    assert n_selections == 784

    # Test edge selections (first and last valid indices)
    features_flat = torch.randn(1, 784)
    feature_mask = torch.zeros(1, 784, dtype=torch.bool)
    selection_mask = torch.tensor([[True]])

    # Test first valid selection (1-indexed)
    afa_selection = torch.tensor([[1]])
    new_mask = direct_unmasker.unmask(
        masked_features=features_flat,
        feature_mask=feature_mask,
        features=features_flat,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )
    assert new_mask[0, 0]  # First feature should be unmasked

    # Test last valid selection (784 in 1-indexed)
    feature_mask = torch.zeros(1, 784, dtype=torch.bool)
    afa_selection = torch.tensor([[784]])
    new_mask = direct_unmasker.unmask(
        masked_features=features_flat,
        feature_mask=feature_mask,
        features=features_flat,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )
    assert new_mask[0, 783]  # Last feature should be unmasked


def test_direct_unmasker_cumulative_selection(
    direct_unmasker: DirectUnmasker,
    single_sample: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test that DirectUnmasker preserves previous selections (cumulative)."""
    single_features, _single_labels = single_sample
    features_flat = single_features.view(1, 784)

    # Start with empty mask
    feature_mask = torch.zeros(1, 784, dtype=torch.bool)

    # First selection
    afa_selection = torch.tensor([[1]])
    selection_mask = torch.tensor([[True]])

    feature_mask = direct_unmasker.unmask(
        masked_features=features_flat * feature_mask.float(),
        feature_mask=feature_mask,
        features=features_flat,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=torch.Size([784]),
    )

    first_unmask_count = torch.sum(feature_mask.int())
    assert first_unmask_count == 1

    # Second selection
    afa_selection = torch.tensor([[100]])

    feature_mask = direct_unmasker.unmask(
        masked_features=features_flat * feature_mask.float(),
        feature_mask=feature_mask,
        features=features_flat,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=torch.Size([784]),
    )

    second_unmask_count = torch.sum(feature_mask.int())

    # Should have two features unmasked now
    assert second_unmask_count == 2
    assert second_unmask_count > first_unmask_count

    # Both features should be unmasked
    assert feature_mask[0, 0]  # First selection
    assert feature_mask[0, 99]  # Second selection
