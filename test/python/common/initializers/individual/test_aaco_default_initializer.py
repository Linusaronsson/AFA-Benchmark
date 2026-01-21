import pytest
import torch

from afabench.common.custom_types import Features
from afabench.common.initializers import AACODefaultInitializer


@pytest.fixture
def features_2d() -> tuple[Features, torch.Size]:
    """2D features for testing."""
    batch_size = 100
    feature_shape = torch.Size([2, 5])
    features = torch.randn(batch_size, *feature_shape)
    return features, feature_shape


def test_aaco_default_basic_functionality(
    features_2d: tuple[Features, torch.Size],
) -> None:
    """Test basic functionality with known dataset."""
    features, feature_shape = features_2d

    kwargs = {"dataset_name": "cubesimple"}
    initializer = AACODefaultInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check that exactly one feature per batch element is selected
    assert mask.sum() == features.shape[0]

    # Check that all batch elements have identical masks (AACO is deterministic)
    mask_flat = mask.view(-1, *feature_shape)
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(mask_flat[0], mask_flat[i])


def test_aaco_default_arbitrary_batch_shape() -> None:
    """Test with arbitrary batch shapes."""
    batch_shape = torch.Size([2, 3])
    feature_shape = torch.Size([4, 5])
    features = torch.randn(*batch_shape, *feature_shape)

    kwargs = {"dataset_name": "cubesimple"}
    initializer = AACODefaultInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check that exactly one feature per batch element is selected
    batch_size = torch.prod(torch.tensor(batch_shape))
    assert mask.sum() == 1 * batch_size

    # Verify all batch elements have identical masks (AACO is deterministic)
    mask_flat = mask.view(-1, *feature_shape)
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(mask_flat[0], mask_flat[i])


def test_aaco_default_known_datasets() -> None:
    """Test with all known dataset configurations."""
    batch_size = 10
    feature_shape = torch.Size([4, 5])  # 20 features total
    features = torch.randn(batch_size, *feature_shape)

    known_datasets = {
        "cube": 6,
        "cubesimple": 3,
        "grid": 1,
        "gas10": 6,
        "mnist": 100,  # This will be clamped to valid range
        "fashionmnist": 100,  # This will be clamped to valid range
        "afacontext": 0,
    }

    for dataset_name, expected_flat_index in known_datasets.items():
        kwargs = {"dataset_name": dataset_name}
        initializer = AACODefaultInitializer(**kwargs)

        if expected_flat_index >= feature_shape.numel():
            # Should raise assertion error for out-of-bounds indices
            with pytest.raises(AssertionError, match="out of bounds"):
                initializer.initialize(
                    features=features, feature_shape=feature_shape
                )
        else:
            mask = initializer.initialize(
                features=features, feature_shape=feature_shape
            )

            # Check that exactly one feature is selected
            assert mask.sum() == batch_size

            # Check that the correct feature is selected
            first_batch_mask = mask[0]
            expected_mask = torch.zeros(feature_shape, dtype=torch.bool)
            multi_index = torch.unravel_index(
                torch.tensor(expected_flat_index, dtype=torch.long),
                feature_shape,
            )
            expected_mask[multi_index] = True
            assert torch.equal(first_batch_mask, expected_mask)


def test_aaco_default_unknown_dataset_fallback() -> None:
    """Test fallback behavior for unknown datasets."""
    batch_size = 15
    feature_shape = torch.Size([6, 7])  # 42 features total
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"dataset_name": "unknown_dataset"}
    initializer = AACODefaultInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Should fall back to middle feature (42 // 2 = 21)
    expected_flat_index = feature_shape.numel() // 2
    expected_mask = torch.zeros(feature_shape, dtype=torch.bool)
    multi_index = torch.unravel_index(
        torch.tensor(expected_flat_index, dtype=torch.long), feature_shape
    )
    expected_mask[multi_index] = True

    # Check that exactly one feature is selected
    assert mask.sum() == batch_size

    # Check that the correct feature is selected
    first_batch_mask = mask[0]
    assert torch.equal(first_batch_mask, expected_mask)


def test_aaco_default_case_insensitive() -> None:
    """Test that dataset names are case insensitive."""
    batch_size = 12
    feature_shape = torch.Size([3, 4])
    features = torch.randn(batch_size, *feature_shape)

    # Test different case variations
    dataset_variations = ["cube", "CUBE", "Cube", "CuBe"]

    masks = []
    for dataset_name in dataset_variations:
        kwargs = {"dataset_name": dataset_name}
        initializer = AACODefaultInitializer(**kwargs)
        mask = initializer.initialize(
            features=features, feature_shape=feature_shape
        )
        masks.append(mask)

    # All should produce identical results
    reference_mask = masks[0]
    for mask in masks[1:]:
        assert torch.equal(reference_mask, mask)


def test_aaco_default_deterministic() -> None:
    """Test that AACODefaultInitializer is deterministic regardless of seed."""
    batch_size = 20
    feature_shape = torch.Size([4, 3])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"dataset_name": "cubesimple"}

    # First run
    initializer1 = AACODefaultInitializer(**kwargs)
    initializer1.set_seed(42)  # Should have no effect
    mask1 = initializer1.initialize(
        features=features, feature_shape=feature_shape
    )

    # Second run with different seed
    initializer2 = AACODefaultInitializer(**kwargs)
    initializer2.set_seed(999)  # Should have no effect
    mask2 = initializer2.initialize(
        features=features, feature_shape=feature_shape
    )

    # Should be identical regardless of seed
    assert torch.equal(mask1, mask2)
    assert mask1.sum() == batch_size
    assert mask2.sum() == batch_size


def test_aaco_default_1d_features() -> None:
    """Test with 1D features."""
    batch_size = 25
    feature_shape = torch.Size([10])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"dataset_name": "cubesimple"}
    initializer = AACODefaultInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == batch_size

    # Feature 3 should be selected for cubesimple
    for i in range(batch_size):
        assert mask[i, 3]  # Feature at index 3
        assert mask[i].sum() == 1


def test_aaco_default_3d_features() -> None:
    """Test with 3D features."""
    batch_size = 8
    feature_shape = torch.Size([2, 3, 4])  # 24 features total
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"dataset_name": "cube"}  # Feature 6
    initializer = AACODefaultInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == batch_size

    # Check that feature 6 maps to correct 3D position
    expected_multi_index = torch.unravel_index(
        torch.tensor(6, dtype=torch.long), feature_shape
    )
    for i in range(batch_size):
        assert mask[i][expected_multi_index]
        assert mask[i].sum() == 1


def test_aaco_default_bounds_checking() -> None:
    """Test bounds checking for dataset configurations."""
    batch_size = 10
    feature_shape = torch.Size([2, 3])  # Only 6 features
    features = torch.randn(batch_size, *feature_shape)

    # MNIST expects feature 100, but we only have 6 features
    kwargs = {"dataset_name": "mnist"}
    initializer = AACODefaultInitializer(**kwargs)

    with pytest.raises(AssertionError, match="out of bounds"):
        initializer.initialize(features=features, feature_shape=feature_shape)


def test_aaco_default_edge_case_single_feature() -> None:
    """Test with single feature space."""
    batch_size = 5
    feature_shape = torch.Size([1])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"dataset_name": "afacontext"}  # Feature 0
    initializer = AACODefaultInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == batch_size
    assert mask.all()  # Single feature should always be True


def test_aaco_default_multidimensional_batch() -> None:
    """Test with complex multi-dimensional batch shapes."""
    batch_shape = torch.Size([2, 3, 4])
    feature_shape = torch.Size([3, 3])  # 9 features total
    features = torch.randn(*batch_shape, *feature_shape)

    kwargs = {"dataset_name": "cubesimple"}  # Feature 3
    initializer = AACODefaultInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape

    # Check that the correct number of features are selected
    total_batch_elements = torch.prod(torch.tensor(batch_shape))
    assert mask.sum() == 1 * total_batch_elements

    # Check that all batch elements have identical patterns
    mask_flat = mask.view(-1, *feature_shape)
    reference_mask = mask_flat[0]
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(reference_mask, mask_flat[i])


def test_aaco_default_consistency() -> None:
    """Test consistency across multiple calls."""
    batch_size = 30
    feature_shape = torch.Size([5, 4])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"dataset_name": "grid"}
    initializer = AACODefaultInitializer(**kwargs)

    # Multiple calls should yield identical results
    masks = []
    for _ in range(5):
        mask = initializer.initialize(
            features=features, feature_shape=feature_shape
        )
        masks.append(mask)

    # All masks should be identical
    reference_mask = masks[0]
    for mask in masks[1:]:
        assert torch.equal(reference_mask, mask)


def test_aaco_default_no_label_dependency() -> None:
    """Test that AACODefaultInitializer doesn't require or use labels."""
    batch_size = 15
    feature_shape = torch.Size([4, 3])
    features = torch.randn(batch_size, *feature_shape)
    labels = torch.randint(0, 2, (batch_size,))

    kwargs = {"dataset_name": "cubesimple"}
    initializer = AACODefaultInitializer(**kwargs)

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


def test_aaco_default_feature_shape_validation() -> None:
    """Test that feature_shape must be provided."""
    batch_size = 10
    feature_shape = torch.Size([3, 3])
    features = torch.randn(batch_size, *feature_shape)

    kwargs = {"dataset_name": "cubesimple"}
    initializer = AACODefaultInitializer(**kwargs)

    with pytest.raises(AssertionError, match="feature_shape must be provided"):
        initializer.initialize(features=features, feature_shape=None)
