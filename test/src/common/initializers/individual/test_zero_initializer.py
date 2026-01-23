import pytest
import torch

from afabench.common.custom_types import Features
from afabench.common.initializers import ZeroInitializer


@pytest.fixture
def features_2d() -> tuple[Features, torch.Size]:
    """2D features for testing."""
    batch_size = 100
    feature_shape = torch.Size([2, 5])
    features = torch.randn(batch_size, *feature_shape)
    return features, feature_shape


def test_zero_initializer_basic_functionality(
    features_2d: tuple[Features, torch.Size],
) -> None:
    """Test basic functionality with 2D features."""
    features, feature_shape = features_2d

    initializer = ZeroInitializer()
    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check that no features are selected
    assert mask.sum() == 0
    assert not mask.any()


def test_zero_initializer_arbitrary_batch_shape() -> None:
    """Test with arbitrary batch shapes."""
    batch_shape = torch.Size([3, 2, 4])
    feature_shape = torch.Size([5, 6])
    features = torch.randn(*batch_shape, *feature_shape)

    initializer = ZeroInitializer()
    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check that no features are selected
    assert mask.sum() == 0
    assert not mask.any()


def test_zero_initializer_1d_features() -> None:
    """Test with 1D features."""
    batch_size = 50
    feature_shape = torch.Size([10])
    features = torch.randn(batch_size, *feature_shape)

    initializer = ZeroInitializer()
    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0
    assert mask.dtype == torch.bool


def test_zero_initializer_3d_features() -> None:
    """Test with 3D features."""
    batch_size = 15
    feature_shape = torch.Size([2, 3, 4])
    features = torch.randn(batch_size, *feature_shape)

    initializer = ZeroInitializer()
    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0
    assert torch.equal(mask, torch.zeros_like(mask))


def test_zero_initializer_deterministic() -> None:
    """Test that ZeroInitializer is always deterministic."""
    batch_size = 20
    feature_shape = torch.Size([4, 3])
    features = torch.randn(batch_size, *feature_shape)

    # First run
    initializer1 = ZeroInitializer()
    initializer1.set_seed(42)  # Should have no effect
    mask1 = initializer1.initialize(
        features=features, feature_shape=feature_shape
    )

    # Second run with different seed
    initializer2 = ZeroInitializer()
    initializer2.set_seed(999)  # Should have no effect
    mask2 = initializer2.initialize(
        features=features, feature_shape=feature_shape
    )

    # Third run with no seed
    initializer3 = ZeroInitializer()
    mask3 = initializer3.initialize(
        features=features, feature_shape=feature_shape
    )

    # All should be identical (all zeros)
    assert torch.equal(mask1, mask2)
    assert torch.equal(mask2, mask3)
    assert mask1.sum() == 0
    assert mask2.sum() == 0
    assert mask3.sum() == 0


def test_zero_initializer_large_features() -> None:
    """Test with large feature spaces."""
    batch_size = 10
    feature_shape = torch.Size([100, 100])  # Large feature space
    features = torch.randn(batch_size, *feature_shape)

    initializer = ZeroInitializer()
    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0
    assert mask.dtype == torch.bool


def test_zero_initializer_single_feature() -> None:
    """Test with single feature."""
    batch_size = 25
    feature_shape = torch.Size([1])
    features = torch.randn(batch_size, *feature_shape)

    initializer = ZeroInitializer()
    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0
    assert not mask[0, 0]  # Single feature should be False


def test_zero_initializer_empty_batch() -> None:
    """Test with empty batch."""
    batch_size = 0
    feature_shape = torch.Size([5, 5])
    features = torch.randn(batch_size, *feature_shape)

    initializer = ZeroInitializer()
    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0
    assert mask.numel() == 0


def test_zero_initializer_multidimensional_batch() -> None:
    """Test with complex multi-dimensional batch shapes."""
    batch_shape = torch.Size([2, 3, 4, 5])
    feature_shape = torch.Size([2, 3])
    features = torch.randn(*batch_shape, *feature_shape)

    initializer = ZeroInitializer()
    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape

    # Check that all elements are zero across all dimensions
    assert mask.sum() == 0
    assert not mask.any()

    # Check shape consistency
    total_elements = torch.prod(torch.tensor(features.shape))
    assert mask.numel() == total_elements


def test_zero_initializer_consistency() -> None:
    """Test consistency across multiple calls."""
    batch_size = 30
    feature_shape = torch.Size([4, 4])
    features = torch.randn(batch_size, *feature_shape)

    initializer = ZeroInitializer()

    # Multiple calls should yield identical results
    masks = []
    for _ in range(5):
        mask = initializer.initialize(
            features=features, feature_shape=feature_shape
        )
        masks.append(mask)

    # All masks should be identical (all zeros)
    reference_mask = masks[0]
    for mask in masks[1:]:
        assert torch.equal(reference_mask, mask)
        assert mask.sum() == 0


def test_zero_initializer_set_seed() -> None:
    """Test that set_seed method exists and is callable."""
    initializer = ZeroInitializer()

    # Should not raise any exceptions
    initializer.set_seed(42)
    initializer.set_seed(None)
    initializer.set_seed(0)
    initializer.set_seed(-1)


def test_zero_initializer_no_label_dependency() -> None:
    """Test that ZeroInitializer doesn't require or use labels."""
    batch_size = 15
    feature_shape = torch.Size([3, 3])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    initializer = ZeroInitializer()

    # Should work with labels
    mask_with_labels = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    # Should work without labels
    mask_without_labels = initializer.initialize(
        features=features, label=None, feature_shape=feature_shape
    )

    # Results should be identical
    assert torch.equal(mask_with_labels, mask_without_labels)
    assert mask_with_labels.sum() == 0
    assert mask_without_labels.sum() == 0


def test_zero_initializer_dtype_consistency() -> None:
    """Test that output dtype is always bool."""
    batch_sizes = [1, 10, 100]
    feature_shapes = [
        torch.Size([1]),
        torch.Size([3, 4]),
        torch.Size([2, 2, 2]),
    ]

    initializer = ZeroInitializer()

    for batch_size in batch_sizes:
        for feature_shape in feature_shapes:
            features = torch.randn(batch_size, *feature_shape)
            mask = initializer.initialize(
                features=features, feature_shape=feature_shape
            )

            assert mask.dtype == torch.bool
            assert mask.sum() == 0
