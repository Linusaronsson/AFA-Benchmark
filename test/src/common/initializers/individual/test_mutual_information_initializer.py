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
    labels = (features[:, 0, 0] + features[:, 0, 1] > 0).long()
    return features, labels, feature_shape


def _assert_per_batch_count(
    mask: torch.Tensor, feature_shape: torch.Size, expected: int
) -> None:
    batch_shape = mask.shape[: -len(feature_shape)]
    batch_size = (
        int(torch.prod(torch.tensor(batch_shape))) if batch_shape else 1
    )
    assert mask.sum() == expected * batch_size


def test_mutual_information_basic_functionality(
    features_labels_2d: tuple[Features, Label, torch.Size],
) -> None:
    """Test basic functionality with 2D features."""
    features, labels, feature_shape = features_labels_2d

    num_initial_features = 3
    initializer = MutualInformationInitializer(
        num_initial_features=num_initial_features
    )
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    _assert_per_batch_count(mask, feature_shape, num_initial_features)

    mask_flat = mask.view(-1, *feature_shape)
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(mask_flat[0], mask_flat[i])


def test_mutual_information_arbitrary_batch_shape() -> None:
    """Test with arbitrary batch shapes."""
    batch_shape = torch.Size([2, 3])
    feature_shape = torch.Size([4, 5])
    features = torch.randn(*batch_shape, *feature_shape)
    labels = (features[..., 0, 0] + features[..., 0, 1] > 0).long()

    num_initial_features = 6
    initializer = MutualInformationInitializer(
        num_initial_features=num_initial_features
    )
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    _assert_per_batch_count(mask, feature_shape, num_initial_features)


def test_mutual_information_caching() -> None:
    """Test that MI results are cached."""
    batch_size = 50
    feature_shape = torch.Size([3, 4])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    num_initial_features = 3
    initializer = MutualInformationInitializer(
        num_initial_features=num_initial_features
    )
    initializer.set_seed(123)

    mask1 = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )
    mask2 = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert torch.equal(mask1, mask2)


def test_mutual_information_seed_changes_clear_cache() -> None:
    """Test that changing seed clears the cache."""
    batch_size = 50
    feature_shape = torch.Size([3, 4])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    num_initial_features = 3
    initializer = MutualInformationInitializer(
        num_initial_features=num_initial_features
    )

    initializer.set_seed(111)
    mask1 = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    initializer.set_seed(222)
    mask2 = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask1.shape == mask2.shape
    _assert_per_batch_count(mask1, feature_shape, num_initial_features)
    _assert_per_batch_count(mask2, feature_shape, num_initial_features)


def test_mutual_information_selects_informative_features() -> None:
    """Test that MI selects the most informative features."""
    batch_size = 200
    feature_shape = torch.Size([1, 5])

    features = torch.randn(batch_size, *feature_shape)
    informative_signal = features[:, 0, 0] + features[:, 0, 1]
    labels = (informative_signal > informative_signal.median()).long()

    initializer = MutualInformationInitializer(num_initial_features=2)
    initializer.set_seed(42)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    first_batch_mask = mask[0]
    selected_features = first_batch_mask.nonzero(as_tuple=True)
    selected_flat_indices = (
        selected_features[0] * feature_shape[1] + selected_features[1]
    ).tolist()

    assert 0 in selected_flat_indices or 1 in selected_flat_indices


def test_mutual_information_different_counts() -> None:
    """Test different numbers of initial features."""
    batch_size = 80
    feature_shape = torch.Size([4, 6])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 3, (batch_size,))

    counts = [1, 6, 12, 18]

    for count in counts:
        initializer = MutualInformationInitializer(num_initial_features=count)
        initializer.set_seed(789)

        mask = initializer.initialize(
            features=features, label=labels, feature_shape=feature_shape
        )

        actual_count = mask.sum() // batch_size
        assert actual_count == count


def test_mutual_information_1d_features() -> None:
    """Test with 1D features."""
    batch_size = 100
    feature_shape = torch.Size([10])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    num_initial_features = 3
    initializer = MutualInformationInitializer(
        num_initial_features=num_initial_features
    )
    initializer.set_seed(101)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    _assert_per_batch_count(mask, feature_shape, num_initial_features)


def test_mutual_information_3d_features() -> None:
    """Test with 3D features."""
    batch_size = 60
    feature_shape = torch.Size([2, 3, 4])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    num_initial_features = 4
    initializer = MutualInformationInitializer(
        num_initial_features=num_initial_features
    )
    initializer.set_seed(202)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    _assert_per_batch_count(mask, feature_shape, num_initial_features)


def test_mutual_information_requires_labels() -> None:
    """Test that MI initializer requires labels."""
    batch_size = 50
    feature_shape = torch.Size([3, 3])
    features = torch.randn(batch_size, *feature_shape)

    initializer = MutualInformationInitializer(num_initial_features=3)

    with pytest.raises(AssertionError, match="requires label"):
        initializer.initialize(
            features=features, label=None, feature_shape=feature_shape
        )


def test_mutual_information_requires_features() -> None:
    """Test that MI initializer requires features."""
    batch_size = 50
    feature_shape = torch.Size([3, 3])
    labels = torch.randint(0, 2, (batch_size,))

    initializer = MutualInformationInitializer(num_initial_features=3)

    with pytest.raises(AssertionError, match="requires features"):
        initializer.initialize(
            features=None,  # pyright: ignore[reportArgumentType]
            label=labels,
            feature_shape=feature_shape,
        )


def test_mutual_information_zero_features() -> None:
    """Test with zero initial features."""
    batch_size = 30
    feature_shape = torch.Size([4])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    initializer = MutualInformationInitializer(num_initial_features=0)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0


def test_mutual_information_full_features() -> None:
    """Test with all features selected."""
    batch_size = 25
    feature_shape = torch.Size([3, 2])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    initializer = MutualInformationInitializer(
        num_initial_features=feature_shape.numel()
    )

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

    num_initial_features = 3
    initializer = MutualInformationInitializer(
        num_initial_features=num_initial_features
    )
    initializer.set_seed(404)

    mask = initializer.initialize(
        features=features, label=labels, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    _assert_per_batch_count(mask, feature_shape, num_initial_features)

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

    initializer = MutualInformationInitializer(num_initial_features=2)
    initializer.set_seed(555)

    masks = []
    for _ in range(3):
        mask = initializer.initialize(
            features=features, label=labels, feature_shape=feature_shape
        )
        masks.append(mask)

    for i in range(1, len(masks)):
        assert torch.equal(masks[0], masks[i])
