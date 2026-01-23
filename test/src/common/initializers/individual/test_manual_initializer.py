import pytest
import torch

from afabench.common.initializers import ManualInitializer


def test_manual_initializer_basic_functionality() -> None:
    """Test basic functionality with simple feature indices."""
    batch_size = 10
    feature_shape = torch.Size([4, 5])
    features = torch.randn(batch_size, *feature_shape)

    # Select features at flat indices 0, 5, 10
    flat_indices = [0, 5, 10]
    kwargs = {"flat_feature_indices": flat_indices}
    initializer = ManualInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check that exactly the right number of features are selected
    assert mask.sum() == len(flat_indices) * batch_size

    # Check that the same features are selected for each batch element
    mask_flat = mask.view(-1, *feature_shape)
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(mask_flat[0], mask_flat[i])


def test_manual_initializer_arbitrary_batch_shape() -> None:
    """Test with arbitrary batch shapes."""
    batch_shape = torch.Size([2, 3])
    feature_shape = torch.Size([3, 4])
    features = torch.randn(*batch_shape, *feature_shape)

    flat_indices = [0, 5, 10]
    kwargs = {"flat_feature_indices": flat_indices}
    initializer = ManualInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    # Check output shape
    assert mask.shape == features.shape

    # Check that correct features are selected
    batch_size = torch.prod(torch.tensor(batch_shape))
    assert mask.sum() == len(flat_indices) * batch_size

    # Verify all batch elements have identical masks (manual is deterministic)
    mask_flat = mask.view(-1, *feature_shape)
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(mask_flat[0], mask_flat[i])


def test_manual_initializer_1d_features() -> None:
    """Test with 1D features."""
    batch_size = 15
    feature_shape = torch.Size([10])
    features = torch.randn(batch_size, *feature_shape)

    flat_indices = [0, 3, 7, 9]
    kwargs = {"flat_feature_indices": flat_indices}
    initializer = ManualInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == len(flat_indices) * batch_size

    # Check that the correct indices are selected
    first_mask = mask[0]
    expected_mask = torch.zeros(feature_shape, dtype=torch.bool)
    expected_mask[flat_indices] = True
    assert torch.equal(first_mask, expected_mask)


def test_manual_initializer_3d_features() -> None:
    """Test with 3D features."""
    batch_size = 8
    feature_shape = torch.Size([2, 3, 4])
    features = torch.randn(batch_size, *feature_shape)

    # Select a few features from the 3D space
    flat_indices = [0, 5, 12, 20]
    kwargs = {"flat_feature_indices": flat_indices}
    initializer = ManualInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == len(flat_indices) * batch_size

    # Verify the correct 3D positions are selected
    first_mask = mask[0]
    expected_mask = torch.zeros(feature_shape, dtype=torch.bool)
    multi_indices = torch.unravel_index(
        torch.tensor(flat_indices, dtype=torch.long), feature_shape
    )
    expected_mask[multi_indices] = True
    assert torch.equal(first_mask, expected_mask)


def test_manual_initializer_single_feature() -> None:
    """Test with a single feature selection."""
    batch_size = 20
    feature_shape = torch.Size([5, 6])
    features = torch.randn(batch_size, *feature_shape)

    flat_indices = [15]  # Just one feature
    kwargs = {"flat_feature_indices": flat_indices}
    initializer = ManualInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 1 * batch_size

    # Check that the correct position is selected
    first_mask = mask[0]
    assert first_mask.sum() == 1

    # Verify it's at the right position
    multi_index = torch.unravel_index(torch.tensor(15), feature_shape)
    assert first_mask[multi_index].item()


def test_manual_initializer_empty_selection() -> None:
    """Test with empty feature selection."""
    batch_size = 10
    feature_shape = torch.Size([4, 4])
    features = torch.randn(batch_size, *feature_shape)

    flat_indices = []  # No features
    kwargs = {"flat_feature_indices": flat_indices}
    initializer = ManualInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == 0


def test_manual_initializer_all_features() -> None:
    """Test selecting all features."""
    batch_size = 5
    feature_shape = torch.Size([3, 3])
    features = torch.randn(batch_size, *feature_shape)

    # Select all features
    num_features = feature_shape.numel()
    flat_indices = list(range(num_features))
    kwargs = {"flat_feature_indices": flat_indices}
    initializer = ManualInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape
    assert mask.sum() == num_features * batch_size
    assert mask.all()  # All features should be True


def test_manual_initializer_deterministic() -> None:
    """Test that ManualInitializer is deterministic regardless of seed."""
    batch_size = 12
    feature_shape = torch.Size([4, 3])
    features = torch.randn(batch_size, *feature_shape)

    flat_indices = [1, 4, 7, 11]
    kwargs = {"flat_feature_indices": flat_indices}

    # First run
    initializer1 = ManualInitializer(**kwargs)
    initializer1.set_seed(42)  # Should have no effect
    mask1 = initializer1.initialize(
        features=features, feature_shape=feature_shape
    )

    # Second run with different seed
    initializer2 = ManualInitializer(**kwargs)
    initializer2.set_seed(999)  # Should have no effect
    mask2 = initializer2.initialize(
        features=features, feature_shape=feature_shape
    )

    # Should be identical regardless of seed
    assert torch.equal(mask1, mask2)


def test_manual_initializer_repr() -> None:
    """Test the string representation."""
    flat_indices = [0, 2, 5, 8]
    kwargs = {"flat_feature_indices": flat_indices}
    initializer = ManualInitializer(**kwargs)

    repr_str = repr(initializer)
    assert "ManualInitializer" in repr_str
    assert str(flat_indices) in repr_str


def test_manual_initializer_bounds_checking() -> None:
    """Test that bounds checking works correctly."""
    batch_size = 10
    feature_shape = torch.Size([3, 4])  # 12 features total
    features = torch.randn(batch_size, *feature_shape)

    # Test with valid indices
    valid_indices = [0, 5, 11]  # All within [0, 12)
    kwargs = {"flat_feature_indices": valid_indices}
    initializer = ManualInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )
    assert mask.shape == features.shape

    # Test with invalid indices (should raise assertion error)
    invalid_indices = [0, 5, 12]  # 12 is out of bounds
    kwargs_invalid = {"flat_feature_indices": invalid_indices}
    initializer_invalid = ManualInitializer(**kwargs_invalid)

    with pytest.raises(AssertionError, match="Feature index out of bounds"):
        initializer_invalid.initialize(
            features=features, feature_shape=feature_shape
        )


def test_manual_initializer_negative_indices() -> None:
    """Test that negative indices are caught."""
    batch_size = 8
    feature_shape = torch.Size([4, 4])
    features = torch.randn(batch_size, *feature_shape)

    negative_indices = [-1, 0, 5]  # -1 is invalid
    kwargs = {"flat_feature_indices": negative_indices}
    initializer = ManualInitializer(**kwargs)

    with pytest.raises(AssertionError, match="Feature index out of bounds"):
        initializer.initialize(features=features, feature_shape=feature_shape)


def test_manual_initializer_multidimensional_batch() -> None:
    """Test with complex multi-dimensional batch shapes."""
    batch_shape = torch.Size([2, 3, 4])
    feature_shape = torch.Size([2, 3])
    features = torch.randn(*batch_shape, *feature_shape)

    flat_indices = [1, 3, 5]
    kwargs = {"flat_feature_indices": flat_indices}
    initializer = ManualInitializer(**kwargs)

    mask = initializer.initialize(
        features=features, feature_shape=feature_shape
    )

    assert mask.shape == features.shape

    # Check that the correct number of features are selected
    total_batch_elements = torch.prod(torch.tensor(batch_shape))
    assert mask.sum() == len(flat_indices) * total_batch_elements

    # Check that all batch elements have identical patterns
    mask_flat = mask.view(-1, *feature_shape)
    reference_mask = mask_flat[0]
    for i in range(1, mask_flat.shape[0]):
        assert torch.equal(reference_mask, mask_flat[i])
