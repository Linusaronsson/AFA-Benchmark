import pytest
import torch

from afabench.common.custom_types import Features
from afabench.common.initializers import FixedRandomInitializer


@pytest.fixture
def features_2d() -> tuple[Features, torch.Size]:
    """2D features for testing."""
    batch_size = 100
    feature_shape = torch.Size([2, 5])
    features = torch.randn(batch_size, *feature_shape)
    return features, feature_shape


def test_fixed_random_basic_functionality(
    features_2d: tuple[Features, torch.Size],
) -> None:
    """Test basic functionality with 2D features."""
    features, feature_shape = features_2d

    kwargs = {"unmask_ratio": 0.3}
    initializer = FixedRandomInitializer(**kwargs)
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check correct number of features selected
    expected_feature_count = int(feature_shape.numel() * 0.3)
    assert mask.sum() == expected_feature_count * features.shape[0]


def test_fixed_random_arbitrary_batch_shape() -> None:
    """Test with arbitrary batch shapes."""
    batch_shape = torch.Size([3, 4, 2])
    feature_shape = torch.Size([5, 6])
    features = torch.randn(*batch_shape, *feature_shape)

    kwargs = {"unmask_ratio": 0.2}
    initializer = FixedRandomInitializer(**kwargs)
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check that all batch elements have the same mask (fixed random)
    expected_count = int(feature_shape.numel() * 0.2)
    batch_size = torch.prod(torch.tensor(batch_shape))
    assert mask.sum() == expected_count * batch_size

    # Verify all batch elements have identical masks
    mask_flat = mask.view(-1, *feature_shape)
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(mask_flat[0], mask_flat[i])


def test_fixed_random_consistency() -> None:
    """Test that the same seed produces consistent results."""
    features = torch.randn(10, 3, 4)
    feature_shape = torch.Size([3, 4])

    kwargs = {"unmask_ratio": 0.25}

    # First run
    initializer1 = FixedRandomInitializer(**kwargs)
    initializer1.set_seed(123)
    mask1 = initializer1.initialize(
        features=features, feature_shape=feature_shape
    )

    # Second run with same seed
    initializer2 = FixedRandomInitializer(**kwargs)
    initializer2.set_seed(123)
    mask2 = initializer2.initialize(
        features=features, feature_shape=feature_shape
    )

    assert torch.equal(mask1, mask2)


def test_fixed_random_caching() -> None:
    """Test that masks are cached and reused."""
    features1 = torch.randn(5, 2, 3)
    features2 = torch.randn(8, 2, 3)  # Different batch size
    feature_shape = torch.Size([2, 3])

    kwargs = {"unmask_ratio": 0.4}
    initializer = FixedRandomInitializer(**kwargs)
    initializer.set_seed(456)

    mask1 = initializer.initialize(
        features=features1, feature_shape=feature_shape
    )
    mask2 = initializer.initialize(
        features=features2, feature_shape=feature_shape
    )

    # Both should use the same underlying pattern, just different batch sizes
    assert mask1[0].equal(mask2[0])  # First batch element should be identical


def test_fixed_random_different_unmask_ratios() -> None:
    """Test different unmask ratios."""
    features = torch.randn(20, 4, 4)
    feature_shape = torch.Size([4, 4])

    ratios = [0.1, 0.25, 0.5, 0.75]

    for ratio in ratios:
        kwargs = {"unmask_ratio": ratio}
        initializer = FixedRandomInitializer(**kwargs)
        initializer.set_seed(789)

        mask = initializer.initialize(
            features=features, feature_shape=feature_shape
        )

        expected_count = int(feature_shape.numel() * ratio)
        actual_count = mask.sum() // features.shape[0]  # Per batch element

        assert actual_count == expected_count


def test_fixed_random_1d_features() -> None:
    """Test with 1D features."""
    batch_size = 50
    feature_shape = torch.Size([10])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"unmask_ratio": 0.3}
    initializer = FixedRandomInitializer(**kwargs)
    initializer.set_seed(101)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == int(feature_shape.numel() * 0.3) * batch_size


def test_fixed_random_3d_features() -> None:
    """Test with 3D features."""
    batch_size = 15
    feature_shape = torch.Size([2, 3, 4])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"unmask_ratio": 0.2}
    initializer = FixedRandomInitializer(**kwargs)
    initializer.set_seed(202)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == int(feature_shape.numel() * 0.2) * batch_size


def test_fixed_random_zero_ratio() -> None:
    """Test with zero unmask ratio."""
    features = torch.randn(10, 5)
    feature_shape = torch.Size([5])

    kwargs = {"unmask_ratio": 0.0}
    initializer = FixedRandomInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0


def test_fixed_random_full_ratio() -> None:
    """Test with full unmask ratio."""
    features = torch.randn(10, 3, 2)
    feature_shape = torch.Size([3, 2])

    kwargs = {"unmask_ratio": 1.0}
    initializer = FixedRandomInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == feature_shape.numel() * features.shape[0]
