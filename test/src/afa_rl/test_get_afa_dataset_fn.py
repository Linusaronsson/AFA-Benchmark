import torch

from afabench.afa_rl.common.dataset_utils import get_afa_dataset_fn


def test_get_afa_dataset_fn() -> None:
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=False)

    features, labels = dataset_fn(
        batch_size=torch.Size((2,)),
        move_on=False,
    )
    assert torch.allclose(
        features,
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    )
    assert torch.allclose(
        labels,
        torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
            ]
        ),
    )

    # We should get the same samples again!
    features, labels = dataset_fn(
        batch_size=torch.Size((2,)),
        move_on=True,
    )
    assert torch.allclose(
        features,
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    )
    assert torch.allclose(
        labels,
        torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
            ]
        ),
    )

    # Now we move on, the dataset should wrap
    features, labels = dataset_fn(
        batch_size=torch.Size((2,)),
        move_on=True,
    )
    assert torch.allclose(
        features,
        torch.tensor(
            [
                [7.0, 8.0, 9.0],
                [1.0, 2.0, 3.0],
            ]
        ),
    )
    assert torch.allclose(
        labels,
        torch.tensor(
            [
                [0, 0, 1],
                [1, 0, 0],
            ]
        ),
    )

    # Get more samples than there are in the dataset
    features, labels = dataset_fn(
        batch_size=torch.Size((6,)),
        move_on=False,
    )
    assert torch.allclose(
        features,
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ),
    )
    assert torch.allclose(
        labels,
        torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        ),
    )


def test_get_afa_dataset_fn_with_2d_images() -> None:
    # Test with 2D grayscale images (3x3)
    all_features = torch.tensor(
        [
            # Sample 1: 3x3 image
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            # Sample 2: 3x3 image
            [
                [10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0],
            ],
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels)

    features, _labels = dataset_fn(
        batch_size=torch.Size((1,)),
        move_on=False,
    )

    # Features should maintain their 2D shape
    expected_features = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ]
    )
    assert torch.allclose(features, expected_features)
    assert features.shape == torch.Size((1, 3, 3))


def test_get_afa_dataset_fn_with_3d_multichannel_images() -> None:
    # Test with 3D multi-channel images (2 channels, 2x2 each)
    all_features = torch.tensor(
        [
            # Sample 1: 2-channel 2x2 image
            [
                [[1.0, 2.0], [3.0, 4.0]],  # Channel 0
                [[5.0, 6.0], [7.0, 8.0]],  # Channel 1
            ],
            # Sample 2: 2-channel 2x2 image
            [
                [[10.0, 11.0], [12.0, 13.0]],  # Channel 0
                [[14.0, 15.0], [16.0, 17.0]],  # Channel 1
            ],
            # Sample 3: 2-channel 2x2 image
            [
                [[20.0, 21.0], [22.0, 23.0]],  # Channel 0
                [[24.0, 25.0], [26.0, 27.0]],  # Channel 1
            ],
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels)

    # Test single sample
    features, labels = dataset_fn(
        batch_size=torch.Size((1,)),
        move_on=False,
    )

    expected_features = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],  # Channel 0
                [[5.0, 6.0], [7.0, 8.0]],  # Channel 1
            ]
        ]
    )
    assert torch.allclose(features, expected_features)
    assert features.shape == torch.Size((1, 2, 2, 2))

    # Test batch of 2 samples
    features, labels = dataset_fn(
        batch_size=torch.Size((2,)),
        move_on=True,
    )

    expected_features = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],  # Sample 1, Channel 0
                [[5.0, 6.0], [7.0, 8.0]],  # Sample 1, Channel 1
            ],
            [
                [[10.0, 11.0], [12.0, 13.0]],  # Sample 2, Channel 0
                [[14.0, 15.0], [16.0, 17.0]],  # Sample 2, Channel 1
            ],
        ]
    )
    assert torch.allclose(features, expected_features)
    assert features.shape == torch.Size((2, 2, 2, 2))

    expected_labels = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
        ]
    )
    assert torch.allclose(labels, expected_labels)


def test_get_afa_dataset_fn_with_4d_features() -> None:
    # Test with 4D features (batch, channels, height, width, depth)
    all_features = torch.tensor(
        [
            # Sample 1: 1 channel, 2x2x2 volume
            [
                [
                    [[1.0, 2.0], [3.0, 4.0]],
                    [[5.0, 6.0], [7.0, 8.0]],
                ]
            ],
            # Sample 2: 1 channel, 2x2x2 volume
            [
                [
                    [[10.0, 11.0], [12.0, 13.0]],
                    [[14.0, 15.0], [16.0, 17.0]],
                ]
            ],
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0],
            [0, 1],
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels)

    features, labels = dataset_fn(
        batch_size=torch.Size((2,)),
        move_on=False,
    )

    # Features should maintain their 4D shape
    assert features.shape == torch.Size((2, 1, 2, 2, 2))
    assert torch.allclose(features, all_features)
    assert torch.allclose(
        labels,
        torch.tensor(
            [
                [1, 0],
                [0, 1],
            ]
        ),
    )


def test_get_afa_dataset_fn_with_shuffling() -> None:
    """Test that shuffling works when enabled."""
    all_features = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    all_labels = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    dataset_fn = get_afa_dataset_fn(all_features, all_labels, shuffle=True)

    # First, move through the dataset to trigger shuffling
    features, labels = dataset_fn(
        batch_size=torch.Size((2,)),
        move_on=True,
    )
    features, labels = dataset_fn(
        batch_size=torch.Size((2,)),
        move_on=True,
    )
    # After this point, the dataset should have been shuffled

    # Get a batch to see the shuffled order
    features, labels = dataset_fn(
        batch_size=torch.Size((3,)),
        move_on=False,
    )

    # We can't predict the exact order due to randomness, but we can verify:
    # 1. All samples are still present
    # 2. Shape is maintained
    assert features.shape == torch.Size((3, 3))
    assert labels.shape == torch.Size((3, 3))

    # Verify all original features are still present (just reordered)
    features_flat = features.view(-1)
    expected_values = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    )
    features_sorted, _ = torch.sort(features_flat)
    expected_sorted, _ = torch.sort(expected_values)
    assert torch.allclose(features_sorted, expected_sorted)
