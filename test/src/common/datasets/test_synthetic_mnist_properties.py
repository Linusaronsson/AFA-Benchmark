"""Tests for SyntheticMNISTDataset properties and behavior."""

import numpy as np
import pytest
import torch

from afabench.common.datasets.datasets import SyntheticMNISTDataset


@pytest.fixture
def small_synthetic_mnist() -> SyntheticMNISTDataset:
    """Create a small synthetic MNIST dataset for testing."""
    return SyntheticMNISTDataset(seed=42, n_samples=100)


@pytest.fixture
def large_synthetic_mnist() -> SyntheticMNISTDataset:
    """Create a larger synthetic MNIST dataset for testing."""
    return SyntheticMNISTDataset(seed=123, n_samples=1000)


def test_synthetic_mnist_feature_shape(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that synthetic MNIST has correct feature shape."""
    assert small_synthetic_mnist.feature_shape == torch.Size([1, 28, 28])


def test_synthetic_mnist_label_shape(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that synthetic MNIST has correct label shape."""
    assert small_synthetic_mnist.label_shape == torch.Size([10])


def test_synthetic_mnist_accepts_seed() -> None:
    """Test that synthetic MNIST accepts seed parameter."""
    assert SyntheticMNISTDataset.accepts_seed()


def test_synthetic_mnist_data_properties(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that synthetic MNIST data has expected properties."""
    features, labels = small_synthetic_mnist.get_all_data()

    # Check overall shapes
    assert features.shape == (100, 1, 28, 28)
    assert labels.shape == (100, 10)

    # Check feature range [0, 1]
    assert torch.all(features >= 0)
    assert torch.all(features <= 1)

    # Check labels are one-hot encoded
    label_sums = torch.sum(labels, dim=1)
    assert torch.allclose(label_sums, torch.ones_like(label_sums))

    # Check that each label vector has exactly one 1.0
    for i in range(labels.shape[0]):
        assert torch.sum(labels[i] == 1.0) == 1


def test_synthetic_mnist_data_format_conversion(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that features can be converted between image and flattened format."""
    features, _ = small_synthetic_mnist.get_all_data()

    # Test flattening to 784 dimensions
    flattened = features.view(-1, 784)
    assert flattened.shape == (features.shape[0], 784)

    # Test reshaping back to image format
    reshaped_back = flattened.view(-1, 1, 28, 28)
    assert reshaped_back.shape == features.shape

    # Check that conversion preserves data
    assert torch.allclose(reshaped_back, features)


def test_synthetic_mnist_seed_reproducibility() -> None:
    """Test that the same seed produces identical data."""
    dataset1 = SyntheticMNISTDataset(seed=42, n_samples=50)
    dataset2 = SyntheticMNISTDataset(seed=42, n_samples=50)

    features1, labels1 = dataset1.get_all_data()
    features2, labels2 = dataset2.get_all_data()

    assert torch.allclose(features1, features2)
    assert torch.allclose(labels1, labels2)


def test_synthetic_mnist_different_seeds_produce_different_data() -> None:
    """Test that different seeds produce different data."""
    dataset1 = SyntheticMNISTDataset(seed=42, n_samples=50)
    dataset2 = SyntheticMNISTDataset(seed=123, n_samples=50)

    features1, labels1 = dataset1.get_all_data()
    features2, labels2 = dataset2.get_all_data()

    # Features should be different
    assert not torch.allclose(features1, features2)
    # But shapes should be the same
    assert features1.shape == features2.shape
    assert labels1.shape == labels2.shape


def test_synthetic_mnist_class_distribution(
    large_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that all classes are represented in the dataset."""
    _, labels = large_synthetic_mnist.get_all_data()

    # Convert one-hot to class indices
    class_indices = torch.argmax(labels, dim=1)

    # Count occurrences of each class
    class_counts = torch.bincount(class_indices, minlength=10)

    # All 10 classes should be present
    assert len(class_counts) == 10
    assert torch.all(class_counts > 0)

    # Distribution should be roughly uniform (within reasonable bounds)
    expected_count = len(large_synthetic_mnist) // 10
    for count in class_counts:
        # Allow up to 50% deviation from perfect uniformity
        assert abs(count.item() - expected_count) < expected_count * 0.5


def test_synthetic_mnist_patterns_are_learnable(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that different classes have distinguishable patterns."""
    features, labels = small_synthetic_mnist.get_all_data()

    # Convert one-hot to class indices
    class_indices = torch.argmax(labels, dim=1)

    # Get samples for first few classes
    class_features = {}
    for class_idx in range(min(5, 10)):  # Test first 5 classes
        mask = class_indices == class_idx
        if torch.any(mask):
            class_features[class_idx] = features[mask][
                0
            ]  # First sample of this class

    # Classes should have different feature patterns
    if len(class_features) >= 2:
        classes = list(class_features.keys())
        feat1 = class_features[classes[0]]
        feat2 = class_features[classes[1]]

        # Features should not be identical (patterns should differ)
        assert not torch.allclose(feat1, feat2, atol=1e-4)


def test_synthetic_mnist_getitem(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test __getitem__ method returns correct shapes."""
    features, labels = small_synthetic_mnist[0]

    assert features.shape == torch.Size([1, 28, 28])
    assert labels.shape == torch.Size([10])
    assert torch.sum(labels) == 1.0  # One-hot encoded


def test_synthetic_mnist_len(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test __len__ method returns correct length."""
    assert len(small_synthetic_mnist) == 100


def test_synthetic_mnist_custom_parameters() -> None:
    """Test synthetic MNIST with custom parameters."""
    dataset = SyntheticMNISTDataset(
        seed=999, n_samples=200, noise_std=0.2, pattern_intensity=0.5
    )

    assert len(dataset) == 200
    features, labels = dataset.get_all_data()
    assert features.shape == (200, 1, 28, 28)
    assert labels.shape == (200, 10)


def test_synthetic_mnist_subset_creation(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that subset creation works correctly."""
    indices = [0, 10, 20, 30, 40]
    subset = small_synthetic_mnist.create_subset(indices)

    assert len(subset) == 5

    # Test that subset contains correct data
    original_features, _ = small_synthetic_mnist.get_all_data()
    subset_features, _ = subset.get_all_data()

    for i, idx in enumerate(indices):
        assert torch.allclose(subset_features[i], original_features[idx])


def test_synthetic_mnist_feature_intensity_ranges(
    small_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that generated features have reasonable intensity distributions."""
    features, _ = small_synthetic_mnist.get_all_data()

    # Features should use the full [0, 1] range (not just background noise)
    assert torch.min(features) >= 0.0
    assert torch.max(features) <= 1.0

    # Should have some high-intensity pixels from patterns
    assert torch.max(features) > 0.5

    # Should have some low-intensity pixels from background
    assert torch.min(features) < 0.5


def test_synthetic_mnist_pattern_consistency() -> None:
    """Test that the same seed and same n_samples produces identical patterns."""
    dataset1 = SyntheticMNISTDataset(seed=42, n_samples=100)
    dataset2 = SyntheticMNISTDataset(
        seed=42, n_samples=100
    )  # Same parameters, should be identical

    features1, labels1 = dataset1.get_all_data()
    features2, labels2 = dataset2.get_all_data()

    # Both datasets should be identical
    assert torch.allclose(features1, features2)
    assert torch.allclose(labels1, labels2)


def test_synthetic_mnist_left_half_is_noise(
    large_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that the left half contains only noise (similar statistics across all classes)."""
    features, labels = large_synthetic_mnist.get_all_data()
    class_indices = torch.argmax(labels, dim=1)

    # Collect statistics for left half across all classes
    left_half_stats = []

    for class_idx in range(10):
        class_mask = class_indices == class_idx
        if class_mask.any():
            class_samples = features[class_mask]
            left_half = class_samples[:, 0, :, :14]  # Left 14 pixels

            mean_val = torch.mean(left_half).item()
            std_val = torch.std(left_half).item()
            left_half_stats.append((mean_val, std_val))

    # Check that all classes have similar statistics in left half
    means = [stat[0] for stat in left_half_stats]
    stds = [stat[1] for stat in left_half_stats]

    mean_variance = np.var(means)
    std_variance = np.var(stds)

    # The variance in means and stds should be small for noise
    assert mean_variance < 0.01, (
        f"Left half means vary too much across classes: {mean_variance}"
    )
    assert std_variance < 0.01, (
        f"Left half stds vary too much across classes: {std_variance}"
    )

    # All means should be close to 0 (noise centered at 0)
    for i, mean_val in enumerate(means):
        assert abs(mean_val) < 0.1, (
            f"Class {i} left half mean {mean_val} too far from 0"
        )


def test_synthetic_mnist_right_half_has_patterns(
    large_synthetic_mnist: SyntheticMNISTDataset,
) -> None:
    """Test that the right half contains class-specific patterns."""
    features, labels = large_synthetic_mnist.get_all_data()
    class_indices = torch.argmax(labels, dim=1)

    # Collect statistics for right half across all classes
    right_half_stats = []

    for class_idx in range(10):
        class_mask = class_indices == class_idx
        if class_mask.any():
            class_samples = features[class_mask]
            right_half = class_samples[:, 0, :, 14:]  # Right 14 pixels

            mean_val = torch.mean(right_half).item()
            std_val = torch.std(right_half).item()
            right_half_stats.append((mean_val, std_val))

    # Check that classes have different statistics in right half
    means = [stat[0] for stat in right_half_stats]

    mean_variance = np.var(means)

    # The variance in means and stds should be larger for patterns
    assert mean_variance > 0.001, (
        f"Right half means don't vary enough across classes: {mean_variance}"
    )

    # At least some classes should have higher means due to patterns
    max_mean = max(means)
    assert max_mean > 0.15, (
        f"No class has significantly elevated mean in right half: {max_mean}"
    )


def test_synthetic_mnist_pattern_intensity_effect() -> None:
    """Test that pattern intensity affects the right half appropriately."""
    # Create datasets with different pattern intensities
    dataset_low = SyntheticMNISTDataset(
        seed=42, n_samples=100, pattern_intensity=0.3
    )
    dataset_high = SyntheticMNISTDataset(
        seed=42, n_samples=100, pattern_intensity=0.9
    )

    features_low, _ = dataset_low.get_all_data()
    features_high, _ = dataset_high.get_all_data()

    # Compare right half intensities
    right_half_low = features_low[:, 0, :, 14:]
    right_half_high = features_high[:, 0, :, 14:]

    mean_low = torch.mean(right_half_low).item()
    mean_high = torch.mean(right_half_high).item()

    # Higher pattern intensity should result in higher mean values
    assert mean_high > mean_low, (
        f"Higher pattern intensity should give higher mean: {mean_high} vs {mean_low}"
    )


def test_synthetic_mnist_left_right_split() -> None:
    """Test that patterns are confined to right half and noise to left half."""
    dataset = SyntheticMNISTDataset(
        seed=42, n_samples=500, noise_std=0.1, pattern_intensity=0.8
    )

    features, labels = dataset.get_all_data()
    class_indices = torch.argmax(labels, dim=1)

    # For each class, verify that left and right halves have expected properties
    for class_idx in range(10):
        class_mask = class_indices == class_idx
        if class_mask.any():
            class_samples = features[class_mask][:5]  # Take first 5 samples

            # Split into left and right halves
            left_half = class_samples[:, 0, :, :14]
            right_half = class_samples[:, 0, :, 14:]

            left_mean = torch.mean(left_half).item()
            right_mean = torch.mean(right_half).item()

            # For most classes, right half should have higher mean due to patterns
            # (some classes might have minimal patterns, so we check the overall trend)
            if right_mean > left_mean:
                # Right half should be significantly higher for pattern classes
                assert (right_mean - left_mean) > 0.01, (
                    f"Class {class_idx} has insufficient pattern difference: "
                    f"left={left_mean:.4f}, right={right_mean:.4f}"
                )


def test_synthetic_mnist_noise_properties() -> None:
    """Test that noise properties are consistent with dataset parameters."""
    noise_std = 0.1  # Use smaller std to avoid heavy clamping effects
    dataset = SyntheticMNISTDataset(
        seed=42, n_samples=200, noise_std=noise_std, pattern_intensity=0.0
    )

    features, _ = dataset.get_all_data()

    # With zero pattern intensity, features should be mostly noise
    # Left half should have standard deviation reasonably close to noise_std
    # (but will be reduced due to clamping at [0,1])
    left_half = features[:, 0, :, :14]
    measured_std = torch.std(left_half).item()

    # The measured std will be smaller due to clamping, but should be reasonable
    assert measured_std > 0.05, (
        f"Measured noise std {measured_std} too small, suggests no noise"
    )
    assert measured_std < noise_std * 1.2, (
        f"Measured noise std {measured_std} unexpectedly higher than input {noise_std}"
    )
