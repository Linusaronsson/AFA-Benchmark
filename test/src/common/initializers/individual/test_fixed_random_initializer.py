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


def _assert_per_batch_count(
    mask: torch.Tensor, feature_shape: torch.Size, expected: int
) -> None:
    batch_shape = mask.shape[: -len(feature_shape)]
    batch_size = (
        int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1
    )
    assert mask.sum() == expected * batch_size


def test_fixed_random_basic_functionality(
    features_2d: tuple[Features, torch.Size],
) -> None:
    """Test basic functionality with 2D features."""
    features, feature_shape = features_2d

    num_initial_features = 3
    initializer = FixedRandomInitializer(
        num_initial_features=num_initial_features
    )
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    _assert_per_batch_count(mask, feature_shape, num_initial_features)


def test_fixed_random_arbitrary_batch_shape() -> None:
    """Test with arbitrary batch shapes."""
    batch_shape = torch.Size([2, 3])
    feature_shape = torch.Size([4, 4])
    features = torch.randn(*batch_shape, *feature_shape)

    num_initial_features = 4
    initializer = FixedRandomInitializer(
        num_initial_features=num_initial_features
    )
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    _assert_per_batch_count(mask, feature_shape, num_initial_features)

    mask_flat = mask.view(-1, *feature_shape)
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(mask_flat[0], mask_flat[i])


def test_fixed_random_consistency() -> None:
    """Test that the same seed produces consistent results."""
    features = torch.randn(10, 3, 4)
    feature_shape = torch.Size([3, 4])

    num_initial_features = 3

    initializer1 = FixedRandomInitializer(
        num_initial_features=num_initial_features
    )
    initializer1.set_seed(123)
    mask1 = initializer1.initialize(
        features=features, feature_shape=feature_shape
    )

    initializer2 = FixedRandomInitializer(
        num_initial_features=num_initial_features
    )
    initializer2.set_seed(123)
    mask2 = initializer2.initialize(
        features=features, feature_shape=feature_shape
    )

    assert torch.equal(mask1, mask2)


def test_fixed_random_different_seeds() -> None:
    """Test that different seeds produce different results."""
    features = torch.randn(20, 4, 3)
    feature_shape = torch.Size([4, 3])

    num_initial_features = 5

    initializer1 = FixedRandomInitializer(
        num_initial_features=num_initial_features
    )
    initializer1.set_seed(111)
    mask1 = initializer1.initialize(
        features=features, feature_shape=feature_shape
    )

    initializer2 = FixedRandomInitializer(
        num_initial_features=num_initial_features
    )
    initializer2.set_seed(222)
    mask2 = initializer2.initialize(
        features=features, feature_shape=feature_shape
    )

    assert not torch.equal(mask1, mask2)


def test_fixed_random_different_counts() -> None:
    """Test different numbers of initial features."""
    features = torch.randn(15, 5, 5)
    feature_shape = torch.Size([5, 5])

    counts = [1, 3, 7, 12]

    for count in counts:
        initializer = FixedRandomInitializer(num_initial_features=count)
        initializer.set_seed(789)

        mask = initializer.initialize(
            features=features, feature_shape=feature_shape
        )

        actual_count = mask.sum() // features.shape[0]
        assert actual_count == count


def test_fixed_random_1d_features() -> None:
    """Test with 1D features."""
    batch_size = 50
    feature_shape = torch.Size([8])
    features = torch.randn(batch_size, *feature_shape)

    num_initial_features = 3
    initializer = FixedRandomInitializer(
        num_initial_features=num_initial_features
    )
    initializer.set_seed(101)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    _assert_per_batch_count(mask, feature_shape, num_initial_features)


def test_fixed_random_3d_features() -> None:
    """Test with 3D features."""
    batch_size = 12
    feature_shape = torch.Size([2, 3, 4])
    features = torch.randn(batch_size, *feature_shape)

    num_initial_features = 4
    initializer = FixedRandomInitializer(
        num_initial_features=num_initial_features
    )
    initializer.set_seed(202)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    _assert_per_batch_count(mask, feature_shape, num_initial_features)


def test_fixed_random_zero_features() -> None:
    """Test with zero initial features."""
    features = torch.randn(10, 6)
    feature_shape = torch.Size([6])

    initializer = FixedRandomInitializer(num_initial_features=0)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0


def test_fixed_random_full_features() -> None:
    """Test with all features selected."""
    features = torch.randn(8, 3, 3)
    feature_shape = torch.Size([3, 3])

    initializer = FixedRandomInitializer(
        num_initial_features=feature_shape.numel()
    )

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == feature_shape.numel() * features.shape[0]
