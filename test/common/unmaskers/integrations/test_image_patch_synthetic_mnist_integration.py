"""Integration tests for ImagePatchUnmasker with synthetic MNIST dataset."""

import pytest
import torch

from afabench.common.datasets.datasets import SyntheticMNISTDataset
from afabench.common.unmaskers.image_patch_unmasker import ImagePatchUnmasker


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
def image_patch_unmasker() -> ImagePatchUnmasker:
    """Create an ImagePatchUnmasker instance for 28x28 -> 4x4."""
    return ImagePatchUnmasker(image_side_length=28, n_channels=1, patch_size=7)


def test_image_patch_unmasker_with_synthetic_mnist_single_sample(
    image_patch_unmasker: ImagePatchUnmasker,
    single_sample: tuple[torch.Tensor, torch.Tensor],
    synthetic_mnist_dataset: SyntheticMNISTDataset,
) -> None:
    """Test ImagePatchUnmasker with single synthetic MNIST sample."""
    single_features, _single_labels = single_sample

    # Check number of selections
    feature_shape = synthetic_mnist_dataset.feature_shape
    n_selections = image_patch_unmasker.get_n_selections(feature_shape)
    assert n_selections == 16  # 4x4 patches

    # Features are already in image format: [batch, channels, height, width]
    features_image = single_features

    # Create initial feature mask (all False)
    feature_mask = torch.zeros(1, 1, 28, 28, dtype=torch.bool)
    masked_features = features_image * feature_mask.float()

    # Test selection of first patch (1-indexed)
    afa_selection = torch.tensor([[1]])  # Select patch 0
    selection_mask = torch.tensor([[True]])

    new_feature_mask = image_patch_unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features_image,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Check output shape
    assert new_feature_mask.shape == (1, 1, 28, 28)

    # Check that a 7x7 patch is unmasked
    patch_sum = torch.sum(new_feature_mask.int())
    assert patch_sum == 7 * 7  # One 7x7 patch should be unmasked


def test_image_patch_unmasker_with_synthetic_mnist_batch(
    image_patch_unmasker: ImagePatchUnmasker,
    batch_samples: tuple[torch.Tensor, torch.Tensor],
    synthetic_mnist_dataset: SyntheticMNISTDataset,
) -> None:
    """Test ImagePatchUnmasker with batch of synthetic MNIST samples."""
    batch_features, _batch_labels = batch_samples
    batch_size = 3
    feature_shape = synthetic_mnist_dataset.feature_shape

    # Features are already in image format
    features_image = batch_features

    # Create initial feature mask (all False)
    feature_mask = torch.zeros(batch_size, 1, 28, 28, dtype=torch.bool)
    masked_features = features_image * feature_mask.float()

    # Test different selections for each sample in batch
    afa_selection = torch.tensor([[1], [2], [3]])  # Different patches
    selection_mask = torch.tensor([[True], [True], [True]])

    new_feature_mask = image_patch_unmasker.unmask(
        masked_features=masked_features,
        feature_mask=feature_mask,
        features=features_image,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=feature_shape,
    )

    # Check output shape
    assert new_feature_mask.shape == (batch_size, 1, 28, 28)

    # Check that each sample has one 7x7 patch unmasked
    for i in range(batch_size):
        patch_sum = torch.sum(new_feature_mask[i].int())
        assert patch_sum == 7 * 7


def test_image_patch_unmasker_feature_shape_mismatch(
    image_patch_unmasker: ImagePatchUnmasker,
    single_sample: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test that ImagePatchUnmasker fails gracefully with wrong feature shape."""
    single_features, _single_labels = single_sample

    # This should reveal the issue - using flattened features with image unmasker
    feature_shape = torch.Size([784])  # Flattened format

    # Create feature mask with wrong shape for image unmasker
    feature_mask = torch.zeros(1, 784, dtype=torch.bool)  # Flattened
    features_flat = single_features.view(1, 784)

    # The unmasker expects image format but gets flattened
    afa_selection = torch.tensor([[1]])
    selection_mask = torch.tensor([[True]])

    with pytest.raises(RuntimeError, match="The size of tensor"):
        # This should fail because feature_mask shape doesn't match
        # what the unmasker expects for image data
        image_patch_unmasker.unmask(
            masked_features=features_flat,
            feature_mask=feature_mask,
            features=features_flat,
            afa_selection=afa_selection,
            selection_mask=selection_mask,
            feature_shape=feature_shape,
        )


def test_cumulative_unmasking(
    image_patch_unmasker: ImagePatchUnmasker,
    single_sample: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Test that unmasking is cumulative (previous masks are preserved)."""
    single_features, _single_labels = single_sample
    features_image = single_features

    # Start with empty mask
    feature_mask = torch.zeros(1, 1, 28, 28, dtype=torch.bool)

    # Unmask first patch
    afa_selection = torch.tensor([[1]])
    selection_mask = torch.tensor([[True]])

    feature_mask = image_patch_unmasker.unmask(
        masked_features=features_image * feature_mask.float(),
        feature_mask=feature_mask,
        features=features_image,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=torch.Size([1, 28, 28]),
    )

    first_unmask_count = torch.sum(feature_mask.int())

    # Unmask second patch
    afa_selection = torch.tensor([[2]])

    feature_mask = image_patch_unmasker.unmask(
        masked_features=features_image * feature_mask.float(),
        feature_mask=feature_mask,
        features=features_image,
        afa_selection=afa_selection,
        selection_mask=selection_mask,
        feature_shape=torch.Size([1, 28, 28]),
    )

    second_unmask_count = torch.sum(feature_mask.int())

    # Should have more unmasked pixels after second patch
    assert second_unmask_count > first_unmask_count
    assert second_unmask_count <= 2 * (7 * 7)  # At most 2 patches
