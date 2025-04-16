import torch

from afa_rl.utils import (
    get_feature_set,
    get_image_feature_set,
    resample_invalid_actions,
)


def test_get_feature_set():
    # First case
    x1 = torch.tensor([3, 0, 2], dtype=torch.float32)
    z1 = torch.tensor([1, 0, 1], dtype=torch.bool)
    s1 = torch.tensor([[3, 1, 0, 0], [0, 0, 0, 0], [2, 0, 0, 1]], dtype=torch.float32)

    # Second case
    x2 = torch.tensor([5, 0, 0], dtype=torch.float32)
    z2 = torch.tensor([1, 1, 0], dtype=torch.bool)
    s2 = torch.tensor([[5, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.float32)

    # Batched, stack them
    x = torch.stack([x1, x2])
    z = torch.stack([z1, z2])
    s = torch.stack([s1, s2])

    s_hat = get_feature_set(x, z)
    assert torch.allclose(s_hat, s)


def test_get_image_feature_set():
    image_shape = (3, 3)
    # Example from docstring (batch size = 1)
    masked_image1 = torch.tensor([3, 0, 1, 2, 1, 0, 0, 2, 3]).float()

    feature_mask1 = torch.tensor([1, 0, 1, 1, 1, 0, 1, 1, 1])
    expected1 = torch.tensor(
        [
            [3, 1, 1],
            [1, 1, 3],
            [2, 2, 1],
            [1, 2, 2],
            [0, 3, 1],
            [2, 3, 2],
            [3, 3, 3],
            [0, 0, 0],  # not observed
            [0, 0, 0],  # not observed
        ]
    ).float()

    # Additional example since it needs to work batched
    masked_image2 = torch.tensor([4, 5, 3, 0, 0, 0, 3, 0, 3]).float()

    feature_mask2 = torch.tensor([1, 1, 1, 0, 1, 0, 1, 0, 1])
    expected2 = torch.tensor(
        [
            [4, 1, 1],
            [5, 1, 2],
            [3, 1, 3],
            [0, 2, 2],
            [3, 3, 1],
            [3, 3, 3],
            [0, 0, 0],  # not observed
            [0, 0, 0],  # not observed
            [0, 0, 0],  # not observed
        ]
    ).float()

    # Stack the two examples
    masked_image = torch.stack([masked_image1, masked_image2])
    feature_mask = torch.stack([feature_mask1, feature_mask2])
    expected = torch.stack([expected1, expected2])

    result = get_image_feature_set(masked_image, feature_mask, image_shape)

    # The order of rows matters because of how nonzero works (row-major)
    assert torch.equal(result, expected), f"\nExpected:\n{expected}\nGot:\n{result}"


def test_resample_invalid_actions():
    # Example action tensor (batch_size)
    actions = torch.tensor([2, 1, 1])

    # Example action mask (batch_size x num_actions)
    action_mask = torch.tensor(
        [
            [False, True, True, False],  # Only indices 1, 2 are valid
            [True, False, False, True],  # Only indices 0, 3 are valid
            [False, False, True, True],  # Only indices 2, 3 are valid
        ],
        dtype=torch.bool,
    )

    # Example action values (batch_size x num_actions)
    action_values = torch.tensor(
        [
            [0.1, 0.2, 0.4, 0.3],  # action 2 is chosen
            [0.3, 0.4, 0.1, 0.2],  # action 0 is chosen
            [0.1, 0.4, 0.2, 0.3],  # action 3 is chosen
        ]
    )

    resampled_actions = resample_invalid_actions(actions, action_mask, action_values)
    assert torch.all(action_mask[torch.arange(actions.shape[0]), resampled_actions])
    # Ensure invalid actions are replaced with the highest valid value
    expected_actions = torch.tensor([2, 0, 3])
    assert torch.all(resampled_actions == expected_actions)


def test_resample_invalid_actions_randomized():
    batch_size = 1000
    num_actions = 10

    # Randomly sample actions
    actions = torch.randint(0, num_actions, (batch_size,))

    # Randomly generate action masks ensuring at least one valid action per row
    action_mask = torch.rand(batch_size, num_actions) > 0.3  # 70% chance of being valid
    valid_mask = action_mask.any(dim=-1)  # Ensure at least one valid action per row
    while not valid_mask.all():  # Regenerate rows with no valid actions
        action_mask[~valid_mask] = torch.rand(1, num_actions) > 0.3
        valid_mask = action_mask.any(dim=-1)

    action_mask = action_mask.to(torch.bool)

    # Random action values
    action_values = torch.rand(batch_size, num_actions)

    # Apply resampling function
    resampled_actions = resample_invalid_actions(actions, action_mask, action_values)

    # Assertions
    assert torch.all(action_mask[torch.arange(batch_size), resampled_actions]), (
        "Some resampled actions are still invalid!"
    )

    # Compute the expected highest valid action per row
    valid_action_values = action_values.clone()
    valid_action_values[~action_mask] = float("-inf")  # Mask invalid actions
    expected_resampled_actions = valid_action_values.argmax(dim=-1)

    # Find original invalid actions
    invalid_mask = ~action_mask[torch.arange(batch_size), actions]

    # Ensure invalid actions were correctly replaced
    assert torch.all(
        resampled_actions[invalid_mask] == expected_resampled_actions[invalid_mask]
    ), "Invalid actions were not replaced with the highest valid action!"
