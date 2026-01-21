import pytest
import torch

from afabench.common.custom_types import Features
from afabench.common.initializers import DynamicRandomInitializer


@pytest.fixture
def features_2d() -> tuple[Features, torch.Size]:
    """2D features for testing."""
    batch_size = 100
    feature_shape = torch.Size([2, 5])
    features = torch.randn(batch_size, *feature_shape)
    return features, feature_shape


def test_dynamic_random_basic_functionality(
    features_2d: tuple[Features, torch.Size],
) -> None:
    """Test basic functionality with 2D features."""
    features, feature_shape = features_2d

    kwargs = {"unmask_ratio": 0.3}
    initializer = DynamicRandomInitializer(**kwargs)
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check correct number of features selected
    expected_feature_count = int(feature_shape.numel() * 0.3)
    assert mask.sum() == expected_feature_count * features.shape[0]


def test_dynamic_random_arbitrary_batch_shape() -> None:
    """Test with arbitrary batch shapes."""
    batch_shape = torch.Size([2, 3])
    feature_shape = torch.Size([4, 4])
    features = torch.randn(*batch_shape, *feature_shape)

    kwargs = {"unmask_ratio": 0.25}
    initializer = DynamicRandomInitializer(**kwargs)
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check that each batch element has correct number of features
    expected_count = int(feature_shape.numel() * 0.25)
    batch_size = torch.prod(torch.tensor(batch_shape))
    assert mask.sum() == expected_count * batch_size

    # Verify that different batch elements have different masks
    mask_flat = mask.view(-1, *feature_shape)
    different_found = False
    for i in range(1, mask_flat.shape[0]):
        if not torch.equal(mask_flat[0], mask_flat[i]):
            different_found = True
            break
    assert different_found, (
        "All batch elements have identical masks, expected different"
    )


def test_dynamic_random_consistency() -> None:
    """Test that the same seed produces consistent results."""
    features = torch.randn(10, 3, 4)
    feature_shape = torch.Size([3, 4])

    kwargs = {"unmask_ratio": 0.25}

    # First run
    initializer1 = DynamicRandomInitializer(**kwargs)
    initializer1.set_seed(123)
    mask1 = initializer1.initialize(
        features=features, feature_shape=feature_shape
    )

    # Second run with same seed
    initializer2 = DynamicRandomInitializer(**kwargs)
    initializer2.set_seed(123)
    mask2 = initializer2.initialize(
        features=features, feature_shape=feature_shape
    )

    assert torch.equal(mask1, mask2)


def test_dynamic_random_different_seeds() -> None:
    """Test that different seeds produce different results."""
    features = torch.randn(20, 4, 3)
    feature_shape = torch.Size([4, 3])

    kwargs = {"unmask_ratio": 0.4}

    # First run with seed 111
    initializer1 = DynamicRandomInitializer(**kwargs)
    initializer1.set_seed(111)
    mask1 = initializer1.initialize(
        features=features, feature_shape=feature_shape
    )

    # Second run with seed 222
    initializer2 = DynamicRandomInitializer(**kwargs)
    initializer2.set_seed(222)
    mask2 = initializer2.initialize(
        features=features, feature_shape=feature_shape
    )

    # Should be different (with very high probability)
    assert not torch.equal(mask1, mask2)


def test_dynamic_random_different_unmask_ratios() -> None:
    """Test different unmask ratios."""
    features = torch.randn(15, 5, 5)
    feature_shape = torch.Size([5, 5])

    ratios = [0.1, 0.25, 0.5, 0.75]

    for ratio in ratios:
        kwargs = {"unmask_ratio": ratio}
        initializer = DynamicRandomInitializer(**kwargs)
        initializer.set_seed(789)

        mask = initializer.initialize(
            features=features, feature_shape=feature_shape
        )

        expected_count = int(feature_shape.numel() * ratio)
        actual_count = mask.sum() // features.shape[0]  # Per batch element

        assert actual_count == expected_count


def test_dynamic_random_1d_features() -> None:
    """Test with 1D features."""
    batch_size = 50
    feature_shape = torch.Size([8])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"unmask_ratio": 0.3}
    initializer = DynamicRandomInitializer(**kwargs)
    initializer.set_seed(101)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == int(feature_shape.numel() * 0.3) * batch_size


def test_dynamic_random_3d_features() -> None:
    """Test with 3D features."""
    batch_size = 12
    feature_shape = torch.Size([2, 3, 4])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"unmask_ratio": 0.2}
    initializer = DynamicRandomInitializer(**kwargs)
    initializer.set_seed(202)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == int(feature_shape.numel() * 0.2) * batch_size


def test_dynamic_random_variability() -> None:
    """Test that masks vary across batch elements."""
    batch_size = 100
    feature_shape = torch.Size([10])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"unmask_ratio": 0.3}
    initializer = DynamicRandomInitializer(**kwargs)
    initializer.set_seed(303)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check that not all batch elements have the same mask
    first_mask = mask[0]
    different_count = 0
    for i in range(1, batch_size):
        if not torch.equal(first_mask, mask[i]):
            different_count += 1

    # At least 90% should be different (probabilistic test)
    assert different_count >= batch_size * 0.9


def test_dynamic_random_zero_ratio() -> None:
    """Test with zero unmask ratio."""
    features = torch.randn(10, 6)
    feature_shape = torch.Size([6])

    kwargs = {"unmask_ratio": 0.0}
    initializer = DynamicRandomInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0


def test_dynamic_random_full_ratio() -> None:
    """Test with full unmask ratio."""
    features = torch.randn(8, 3, 3)
    feature_shape = torch.Size([3, 3])

    kwargs = {"unmask_ratio": 1.0}
    initializer = DynamicRandomInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == feature_shape.numel() * features.shape[0]


def test_dynamic_random_multidimensional_batch() -> None:
    """Test with complex multi-dimensional batch shapes."""
    batch_shape = torch.Size([3, 2, 4])
    feature_shape = torch.Size([2, 2])
    features = torch.randn(*batch_shape, *feature_shape)

    kwargs = {"unmask_ratio": 0.5}
    initializer = DynamicRandomInitializer(**kwargs)
    initializer.set_seed(404)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape

    # Check that each batch element has the right number of features
    expected_per_element = int(feature_shape.numel() * 0.5)
    total_batch_elements = torch.prod(torch.tensor(batch_shape))
    assert mask.sum() == expected_per_element * total_batch_elements

    # Check variability across flattened batch dimension
    mask_flat = mask.view(-1, *feature_shape)
    different_found = False
    for i in range(1, min(10, mask_flat.shape[0])):  # Check first 10
        if not torch.equal(mask_flat[0], mask_flat[i]):
            different_found = True
            break
    assert different_found, "Expected variation in masks across batch elements"
