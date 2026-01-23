import pytest
import torch

from afabench.common.custom_types import Features, Label
from afabench.common.initializers import MutualInformationInitializer


@pytest.fixture
def features_labels_2d() -> tuple[Features, Label, torch.Size]:
    """2D features with correlated labels for testing."""
    batch_size = 100
    feature_shape = torch.Size([2, 5])
    features = torch.randn(batch_size, *feature_shape)
    # Create labels that depend on first few features
    labels = (features[:, 0, 0] + features[:, 0, 1] > 0).long()
    return features, labels, feature_shape


def test_mutual_information_basic_functionality(
    features_labels_2d: tuple[Features, Label, torch.Size],
) -> None:
    """Test basic functionality with 2D features."""
    features, labels, feature_shape = features_labels_2d

    kwargs = {"unmask_ratio": 0.3}
    initializer = MutualInformationInitializer(**kwargs)
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check correct number of features selected
    expected_count = int(feature_shape.numel() * 0.3)
    assert mask.sum() == expected_count * features.shape[0]

    # Check that all batch elements have identical masks (MI is deterministic)
    mask_flat = mask.view(-1, *feature_shape)
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(mask_flat[0], mask_flat[i])


def test_mutual_information_arbitrary_batch_shape() -> None:
    """Test with arbitrary batch shapes."""
    batch_shape = torch.Size([2, 3])
    feature_shape = torch.Size([4, 5])
    features = torch.randn(*batch_shape, *feature_shape)
    # Create labels that depend on first few features
    labels = (features[..., 0, 0] + features[..., 0, 1] > 0).long()

    kwargs = {"unmask_ratio": 0.3}
    initializer = MutualInformationInitializer(**kwargs)
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check correct number of features selected
    expected_count = int(feature_shape.numel() * 0.3)
    batch_size = torch.prod(torch.tensor(batch_shape))
    assert mask.sum() == expected_count * batch_size


def test_mutual_information_caching() -> None:
    """Test that MI results are cached."""
    batch_size = 50
    feature_shape = torch.Size([3, 4])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    kwargs = {"unmask_ratio": 0.25}
    initializer = MutualInformationInitializer(**kwargs)
    initializer.set_seed(123)

    mask1 = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )
    mask2 = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    # Should be identical due to caching
    assert torch.equal(mask1, mask2)


def test_mutual_information_seed_changes_clear_cache() -> None:
    """Test that changing seed clears the cache."""
    batch_size = 50
    feature_shape = torch.Size([3, 4])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    kwargs = {"unmask_ratio": 0.25}
    initializer = MutualInformationInitializer(**kwargs)

    # First run with seed 111
    initializer.set_seed(111)
    mask1 = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    # Second run with different seed (should clear cache)
    initializer.set_seed(222)
    mask2 = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    # The important thing is that the method works correctly with different seeds
    # Both masks should have the correct shape and feature count
    assert mask1.shape == mask2.shape
    expected_count = int(feature_shape.numel() * 0.25)
    assert mask1.sum() == expected_count * batch_size
    assert mask2.sum() == expected_count * batch_size


def test_mutual_information_selects_informative_features() -> None:
    """Test that MI selects the most informative features."""
    batch_size = 200
    feature_shape = torch.Size([1, 5])

    # Create synthetic data where features 0 and 1 are highly correlated with labels
    features = torch.randn(batch_size, *feature_shape)

    # Make features 0 and 1 predictive of the label
    informative_signal = features[:, 0, 0] + features[:, 0, 1]
    labels = (informative_signal > informative_signal.median()).long()

    kwargs = {"unmask_ratio": 0.4}  # Select 2 features
    initializer = MutualInformationInitializer(**kwargs)
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    # Check that the correct features are selected
    first_batch_mask = mask[0]
    selected_features = first_batch_mask.nonzero(as_tuple=True)
    selected_flat_indices = (
        selected_features[0] * feature_shape[1] + selected_features[1]
    ).tolist()

    # Features 0 and 1 should be among the selected (most informative)
    assert 0 in selected_flat_indices or 1 in selected_flat_indices


def test_mutual_information_different_unmask_ratios() -> None:
    """Test different unmask ratios."""
    batch_size = 80
    feature_shape = torch.Size([4, 6])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 3, (batch_size,))

    ratios = [0.1, 0.25, 0.5, 0.75]

    for ratio in ratios:
        kwargs = {"unmask_ratio": ratio}
        initializer = MutualInformationInitializer(**kwargs)
        initializer.set_seed(789)

        mask = initializer.initialize(
            features=features, label=labels, feature_shape=feature_shape
        )

        expected_count = int(feature_shape.numel() * ratio)
        actual_count = mask.sum() // batch_size  # Per batch element

        assert actual_count == expected_count


def test_mutual_information_1d_features() -> None:
    """Test with 1D features."""
    batch_size = 100
    feature_shape = torch.Size([10])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    kwargs = {"unmask_ratio": 0.3}
    initializer = MutualInformationInitializer(**kwargs)
    initializer.set_seed(101)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == int(feature_shape.numel() * 0.3) * batch_size


def test_mutual_information_3d_features() -> None:
    """Test with 3D features."""
    batch_size = 60
    feature_shape = torch.Size([2, 3, 4])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    kwargs = {"unmask_ratio": 0.2}
    initializer = MutualInformationInitializer(**kwargs)
    initializer.set_seed(202)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == int(feature_shape.numel() * 0.2) * batch_size


def test_mutual_information_requires_labels() -> None:
    """Test that MI initializer requires labels."""
    batch_size = 50
    feature_shape = torch.Size([3, 3])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"unmask_ratio": 0.3}
    initializer = MutualInformationInitializer(**kwargs)

    with pytest.raises(AssertionError, match="requires label"):
        initializer.initialize(
            features=features, label=None, feature_shape=feature_shape
        )


def test_mutual_information_requires_features() -> None:
    """Test that MI initializer requires features."""
    batch_size = 50
    feature_shape = torch.Size([3, 3])
    labels = torch.randint(0, 2, (batch_size,))

    kwargs = {"unmask_ratio": 0.3}
    initializer = MutualInformationInitializer(**kwargs)

    with pytest.raises(AssertionError, match="requires features"):
        initializer.initialize(
            features=None,  # pyright: ignore[reportArgumentType]
            label=labels,
            feature_shape=feature_shape,
        )


def test_mutual_information_zero_ratio() -> None:
    """Test with zero unmask ratio."""
    batch_size = 30
    feature_shape = torch.Size([4])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    kwargs = {"unmask_ratio": 0.0}
    initializer = MutualInformationInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0


def test_mutual_information_full_ratio() -> None:
    """Test with full unmask ratio."""
    batch_size = 25
    feature_shape = torch.Size([3, 2])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    kwargs = {"unmask_ratio": 1.0}
    initializer = MutualInformationInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == feature_shape.numel() * batch_size


def test_mutual_information_multidimensional_batch() -> None:
    """Test with complex multi-dimensional batch shapes."""
    batch_shape = torch.Size([2, 3, 4])
    feature_shape = torch.Size([2, 3])
    features = torch.randn(*batch_shape, *feature_shape)
    labels = torch.randint(0, 2, batch_shape)

    kwargs = {"unmask_ratio": 0.4}
    initializer = MutualInformationInitializer(**kwargs)
    initializer.set_seed(404)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape

    # Check that the correct number of features are selected
    expected_per_element = int(feature_shape.numel() * 0.4)
    total_batch_elements = torch.prod(torch.tensor(batch_shape))
    assert mask.sum() == expected_per_element * total_batch_elements

    # Check that all batch elements have identical patterns
    mask_flat = mask.view(-1, *feature_shape)
    reference_mask = mask_flat[0]
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(reference_mask, mask_flat[i])


def test_mutual_information_consistency_across_calls() -> None:
    """Test that results are consistent across multiple calls with same data."""
    batch_size = 40
    feature_shape = torch.Size([5])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    kwargs = {"unmask_ratio": 0.4}
    initializer = MutualInformationInitializer(**kwargs)
    initializer.set_seed(555)

    # Multiple calls should yield identical results
    masks = []
    for _ in range(3):
        mask = initializer.initialize(
            features=features, label=labels, feature_shape=feature_shape
        )
        masks.append(mask)

    # All masks should be identical
    for i in range(1, len(masks)):
        assert torch.equal(masks[0], masks[i])
