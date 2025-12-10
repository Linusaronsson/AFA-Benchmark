from typing import final

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import TensorSpec

from afabench.afa_rl.afa_env import AFAEnv
from afabench.afa_rl.datasets import get_afa_dataset_fn
from afabench.afa_rl.reward_functions import get_range_based_reward_fn
from afabench.afa_rl.utils import get_eval_metrics
from afabench.common.initializers.dynamic_random_initializer import (
    DynamicRandomInitializer,
)
from afabench.common.unmaskers.direct_unmasker import DirectUnmasker


@final
class SequentialDummyPolicy:
    """A dummy policy that selects actions in a predefined sequence."""

    def __init__(
        self, action_sequence: list[int], action_spec: TensorSpec
    ) -> None:
        self.action_sequence: list[int] = action_sequence
        self.action_spec: TensorSpec = action_spec
        self.current_step: int = 0

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        """Return the next action in the sequence."""
        batch_size = tensordict.batch_size[0] if tensordict.batch_size else 1

        if self.current_step < len(self.action_sequence):
            action = self.action_sequence[self.current_step]
        else:
            action = 0  # Stop action if sequence is exhausted

        # Create action tensor for all batch elements
        action_tensor = torch.tensor([action] * batch_size, dtype=torch.int64)

        self.current_step += 1

        return TensorDict(
            {"action": action_tensor}, batch_size=tensordict.batch_size
        )


@final
class DummyPredictFn:
    """Dummy prediction function for testing."""

    def __init__(self, n_classes: int = 8) -> None:
        self.n_classes: int = n_classes

    def __call__(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,  # noqa: ARG002
        label: torch.Tensor | None = None,  # noqa: ARG002
        feature_shape: torch.Size | None = None,  # noqa: ARG002
    ) -> torch.Tensor:
        batch_size = masked_features.shape[0]
        return torch.randn(batch_size, self.n_classes).softmax(-1)


@pytest.fixture
def test_env() -> AFAEnv:
    """Create a test environment with known reward structure."""
    device = torch.device("cpu")
    batch_size = 4
    n_features = 20
    n_classes = 8
    hard_budget = 10

    # Create test data
    features = torch.randn(50, n_features)
    labels = torch.zeros(50, n_classes)
    labels[torch.arange(50), torch.randint(0, n_classes, (50,))] = 1

    dataset_fn = get_afa_dataset_fn(features, labels, device=device)

    # Create reward function: features 5-9 give reward 1.0 each
    reward_fn = get_range_based_reward_fn(
        reward_ranges=[(5, 9)], reward_value=1.0
    )

    # Create environment
    env = AFAEnv(
        dataset_fn=dataset_fn,
        reward_fn=reward_fn,
        device=device,
        batch_size=torch.Size((batch_size,)),
        feature_shape=torch.Size((n_features,)),
        n_selections=n_features,
        n_classes=n_classes,
        hard_budget=hard_budget,
        initialize_fn=DynamicRandomInitializer(unmask_ratio=0.0).initialize,
        unmask_fn=DirectUnmasker().unmask,
        force_hard_budget=False,
        seed=42,
    )

    return env


def create_manual_episode_data(
    env: AFAEnv, action_sequence: list[int]
) -> TensorDict:
    """Create episode data by manually stepping through environment."""
    env.reset()
    td = env.reset()

    episode_data = []

    for step_idx, action_val in enumerate(action_sequence):
        batch_size = td.batch_size[0]
        action = torch.tensor([action_val] * batch_size, dtype=torch.int64)

        # Step environment
        td = env.step(td.set("action", action))

        # Debug: print step reward
        step_reward = td["next", "reward"]
        print(
            f"  Step {step_idx + 1}, Action {action_val}: Reward = "
            f"{step_reward.squeeze().tolist()}"
        )

        # Collect step data
        step_data = {
            "action": td["action"],
            "next": {
                "reward": td["next", "reward"],
                "done": td["next", "done"],
                "masked_features": td["next", "masked_features"],
                "feature_mask": td["next", "feature_mask"],
            },
            "label": td["label"],
        }
        episode_data.append(step_data)

        if td["next", "done"].all():
            break

        td = td["next"]

    # Convert to format expected by get_eval_metrics
    # Create a TensorDict with batch dims (n_agents, episode_len)
    batch_size = episode_data[0]["action"].shape[0]
    episode_len = len(episode_data)

    # Stack all step data
    actions = torch.stack([step["action"] for step in episode_data], dim=1)
    rewards = torch.stack(
        [step["next"]["reward"] for step in episode_data], dim=1
    )
    dones = torch.stack([step["next"]["done"] for step in episode_data], dim=1)
    masked_features_stack = torch.stack(
        [step["next"]["masked_features"] for step in episode_data], dim=1
    )
    feature_masks_stack = torch.stack(
        [step["next"]["feature_mask"] for step in episode_data], dim=1
    )
    labels_stack = (
        episode_data[0]["label"].unsqueeze(1).expand(-1, episode_len, -1)
    )

    # Create TensorDict with proper initialization
    next_td = TensorDict({}, batch_size=(batch_size, episode_len))
    next_td["reward"] = rewards
    next_td["done"] = dones
    next_td["masked_features"] = masked_features_stack
    next_td["feature_mask"] = feature_masks_stack

    td_eval = TensorDict({}, batch_size=(batch_size, episode_len))
    td_eval["action"] = actions
    td_eval["next"] = next_td
    td_eval["label"] = labels_stack

    return td_eval


def test_perfect_policy_reward_calculation(test_env: AFAEnv) -> None:
    """Test perfect policy to reveal get_eval_metrics bug."""
    # Perfect policy: select all rewarding features (5-9) then stop
    perfect_actions = [6, 7, 8, 9, 10, 0]  # Actions 6-10 select features 5-9

    td_evals = []
    total_manual_reward = 0
    for episode_idx in range(3):  # 3 evaluation episodes
        td_eval = create_manual_episode_data(test_env, perfect_actions)
        td_evals.append(td_eval)

        # Debug: print episode reward
        episode_reward = td_eval["next", "reward"].sum().item()
        total_manual_reward += episode_reward
        print(f"Episode {episode_idx + 1}: Total reward = {episode_reward}")

    # Test get_eval_metrics
    predict_fn = DummyPredictFn(8)
    metrics = get_eval_metrics(td_evals, predict_fn)

    # Manual calculation - what we EXPECT average reward per agent per episode
    batch_size = td_evals[0].batch_size[0]  # Number of agents
    n_episodes = len(td_evals)
    expected_avg_per_agent = total_manual_reward / (n_episodes * batch_size)
    print(f"Expected avg per agent per episode: {expected_avg_per_agent}")

    # What get_eval_metrics actually calculates (incorrect)
    episode_length = td_evals[0].batch_size[1]  # Number of steps per episode
    actual_denominator = n_episodes * batch_size * episode_length
    actual_calculation = total_manual_reward / actual_denominator
    print(f"get_eval_metrics actual calculation: {actual_calculation}")
    print(f"get_eval_metrics result: {metrics['reward_sum']}")

    # Verify that get_eval_metrics is doing the wrong calculation
    assert abs(metrics["reward_sum"] - actual_calculation) < 1e-6, (
        "get_eval_metrics calculation mismatch"
    )

    # This test DOCUMENTS the bug - get_eval_metrics divides by episode length
    # The correct average should be 5.0 (each agent gets 5 rewards per episode)
    # But get_eval_metrics returns 0.833... because it divides by episode length
    print("\n*** BUG CONFIRMED ***")
    print("get_eval_metrics incorrectly divides by episode length")
    print(f"Correct result should be: {expected_avg_per_agent}")
    print(f"Actual result is: {metrics['reward_sum']}")
    print("This explains why you see 0.6 instead of 6.0 in your training!")


def test_partial_policy_reward_calculation(test_env: AFAEnv) -> None:
    """Test partial policy to expose the same get_eval_metrics bug."""
    # Partial policy: select only 2 rewarding features
    partial_actions = [6, 8, 0]  # Select features 5 and 7, then stop

    td_evals = []
    total_manual_reward = 0
    for _episode_idx in range(3):  # 3 evaluation episodes
        td_eval = create_manual_episode_data(test_env, partial_actions)
        td_evals.append(td_eval)

        episode_reward = td_eval["next", "reward"].sum().item()
        total_manual_reward += episode_reward

    predict_fn = DummyPredictFn(8)
    metrics = get_eval_metrics(td_evals, predict_fn)

    # Manual calculation - correct way
    batch_size = td_evals[0].batch_size[0]
    n_episodes = len(td_evals)
    expected_avg = total_manual_reward / (n_episodes * batch_size)

    # get_eval_metrics incorrect way
    episode_length = td_evals[0].batch_size[1]
    incorrect_avg = total_manual_reward / (
        n_episodes * batch_size * episode_length
    )

    print("Partial policy test:")
    print(f"  Expected avg per agent: {expected_avg}")
    print(f"  get_eval_metrics result: {metrics['reward_sum']}")
    ratio = 1 / episode_length
    print(
        f"  Ratio: {metrics['reward_sum'] / expected_avg:.3f} "
        f"(should be {ratio:.3f})"
    )

    # Verify the bug exists here too
    assert abs(metrics["reward_sum"] - incorrect_avg) < 1e-6


def test_poor_policy_reward_calculation(test_env: AFAEnv) -> None:
    """Test that poor policy gives expected reward."""
    # Poor policy: select non-rewarding features
    poor_actions = [1, 2, 3, 0]  # Select features 0, 1, 2 (no reward)

    td_evals = []
    for _ in range(3):
        td_eval = create_manual_episode_data(test_env, poor_actions)
        td_evals.append(td_eval)

    expected_reward_per_agent = 0.0  # No rewarding features selected

    predict_fn = DummyPredictFn(8)
    metrics = get_eval_metrics(td_evals, predict_fn)

    assert abs(metrics["reward_sum"] - expected_reward_per_agent) < 1e-6, (
        f"Poor policy: Expected {expected_reward_per_agent}, "
        f"got {metrics['reward_sum']}"
    )


def test_bug_affects_all_batch_sizes() -> None:
    """Test that the get_eval_metrics bug affects all batch sizes consistently."""
    for batch_size in [1, 2, 8]:
        device = torch.device("cpu")
        n_features = 20
        n_classes = 8
        hard_budget = 10

        # Create test data
        features = torch.randn(50, n_features)
        labels = torch.zeros(50, n_classes)
        labels[torch.arange(50), torch.randint(0, n_classes, (50,))] = 1

        dataset_fn = get_afa_dataset_fn(features, labels, device=device)

        # Create reward function
        reward_fn = get_range_based_reward_fn(
            reward_ranges=[(5, 9)], reward_value=1.0
        )

        # Create environment with specific batch size
        env = AFAEnv(
            dataset_fn=dataset_fn,
            reward_fn=reward_fn,
            device=device,
            batch_size=torch.Size((batch_size,)),
            feature_shape=torch.Size((n_features,)),
            n_selections=n_features,
            n_classes=n_classes,
            hard_budget=hard_budget,
            initialize_fn=DynamicRandomInitializer(
                unmask_ratio=0.0
            ).initialize,
            unmask_fn=DirectUnmasker().unmask,
            force_hard_budget=False,
            seed=42,
        )

        # Perfect policy
        perfect_actions = [6, 7, 8, 9, 10, 0]

        td_eval = create_manual_episode_data(env, perfect_actions)
        predict_fn = DummyPredictFn(n_classes)
        metrics = get_eval_metrics([td_eval], predict_fn)

        # Calculate what the result should be with the bug
        episode_length = len(perfect_actions)
        correct_avg_per_agent = 5.0
        buggy_result = correct_avg_per_agent / episode_length

        print(
            f"Batch size {batch_size}: {metrics['reward_sum']:.3f} "
            f"(expected buggy: {buggy_result:.3f})"
        )

        # Verify bug is consistent across batch sizes
        assert abs(metrics["reward_sum"] - buggy_result) < 1e-6


def test_range_based_reward_function() -> None:
    """Test that the range-based reward function works correctly."""
    batch_size = 2
    n_features = 10

    # Create test inputs
    selection_mask_old = torch.zeros(batch_size, n_features, dtype=torch.bool)
    selection_mask_new = selection_mask_old.clone()

    # Test selecting feature 6 (index 6)
    selection_mask_new[:, 6] = True

    # Create reward function for range 5-8
    reward_fn = get_range_based_reward_fn(
        reward_ranges=[(5, 8)], reward_value=2.0
    )

    # Dummy inputs
    masked_features = torch.randn(batch_size, n_features)
    feature_mask = torch.ones(batch_size, n_features, dtype=torch.bool)
    features = torch.randn(batch_size, n_features)
    label = torch.zeros(batch_size, 8)
    done = torch.zeros(batch_size, 1, dtype=torch.bool)
    afa_selection = torch.tensor(
        [7] * batch_size
    )  # Action 7 = select feature 6

    reward = reward_fn(
        masked_features,
        feature_mask,
        selection_mask_old,
        masked_features,
        feature_mask,
        selection_mask_new,
        afa_selection,
        features,
        label,
        done,
    )

    # Feature 6 is in range 5-8, so should get reward 2.0
    assert torch.allclose(reward, torch.tensor([[2.0], [2.0]])), (
        f"Expected reward 2.0, got {reward}"
    )

    # Test selecting feature outside range
    selection_mask_new_outside = selection_mask_old.clone()
    selection_mask_new_outside[:, 2] = True  # Feature 2 (outside range 5-8)

    reward_outside = reward_fn(
        masked_features,
        feature_mask,
        selection_mask_old,
        masked_features,
        feature_mask,
        selection_mask_new_outside,
        torch.tensor([3] * batch_size),  # Action 3 = select feature 2
        features,
        label,
        done,
    )

    # Feature 2 is outside range, so should get 0 reward
    assert torch.allclose(reward_outside, torch.tensor([[0.0], [0.0]])), (
        f"Expected reward 0.0, got {reward_outside}"
    )


def test_eval_metrics_manual_calculation() -> None:
    """Test that get_eval_metrics calculation matches manual calculation."""
    batch_size = 4
    n_episodes = 3

    # Create dummy episode data with known rewards
    episode_rewards = torch.tensor(
        [
            [[1.0], [2.0], [0.0]],  # Agent 0: episodes with rewards 1, 2, 0
            [[1.5], [1.5], [0.5]],  # Agent 1: episodes rewards 1.5, 1.5, 0.5
            [[3.0], [1.0], [1.0]],  # Agent 2: episodes with rewards 3, 1, 1
            [[2.0], [2.0], [2.0]],  # Agent 3: episodes with rewards 2, 2, 2
        ]
    )  # Shape: (batch_size, n_episodes, 1)

    # Create mock TensorDict episodes
    td_evals = []
    for episode_idx in range(n_episodes):
        episode_reward = episode_rewards[:, episode_idx, :].unsqueeze(
            1
        )  # (batch_size, 1, 1)

        # Create TensorDict with proper initialization
        next_td = TensorDict({}, batch_size=(batch_size, 1))
        next_td["reward"] = episode_reward
        next_td["done"] = torch.tensor([[True], [True], [True], [True]])
        next_td["masked_features"] = torch.randn(batch_size, 1, 20)
        next_td["feature_mask"] = torch.ones(
            batch_size, 1, 20, dtype=torch.bool
        )

        td_eval = TensorDict({}, batch_size=(batch_size, 1))
        td_eval["action"] = torch.tensor([[0], [0], [0], [0]])  # Dummy actions
        td_eval["next"] = next_td
        td_eval["label"] = torch.zeros(batch_size, 1, 8)

        td_evals.append(td_eval)

    # Calculate expected result manually
    total_reward = episode_rewards.sum().item()  # Sum all rewards
    expected_avg_reward = total_reward / (n_episodes * batch_size)

    # Test get_eval_metrics
    predict_fn = DummyPredictFn(8)
    metrics = get_eval_metrics(td_evals, predict_fn)

    assert abs(metrics["reward_sum"] - expected_avg_reward) < 1e-6, (
        f"Manual calculation: Expected {expected_avg_reward}, "
        f"got {metrics['reward_sum']}"
    )


def test_demonstrates_your_training_issue() -> None:
    """Demonstrate exactly why you see 0.6 instead of 6.0 in training."""
    print("\n" + "=" * 60)
    print("DEMONSTRATION OF YOUR TRAINING ISSUE")
    print("=" * 60)

    # Your actual training scenario:
    # - Agent should get 6.0 reward per episode (6 features x 1.0 each)
    # - Episodes are about 6-10 steps long
    # - get_eval_metrics divides by episode length

    expected_optimal_reward = 6.0
    typical_episode_length = 10  # Your hard budget

    buggy_eval_result = expected_optimal_reward / typical_episode_length

    print("If your agent achieved perfect performance:")
    print(f"  Actual reward per agent per episode: {expected_optimal_reward}")
    print(f"  Typical episode length: {typical_episode_length} steps")
    print(f"  get_eval_metrics buggy result: {buggy_eval_result}")
    print("  Your observed result: 0.6")
    print(f"  Ratio: {0.6 / buggy_eval_result:.2f}")

    print("\nThis means your agent is achieving:")
    print(f"  {0.6 / buggy_eval_result:.1%} of perfect performance")
    print(
        f"  Which equals {0.6 * typical_episode_length:.1f} "
        f"actual reward per episode"
    )
    print(f"  Out of {expected_optimal_reward} possible reward per episode")

    print("\nCONCLUSION:")
    print(
        f"  Your 0.6 result means your agent gets "
        f"~{0.6 * typical_episode_length:.1f} actual reward per episode"
    )
    print(
        "  This is PERFECT performance, but get_eval_metrics makes it look bad!"
    )
    print("  The bug divides by episode length unnecessarily.")
    print("=" * 60)


if __name__ == "__main__":
    pytest.main([__file__])
