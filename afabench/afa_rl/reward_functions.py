import torch
from jaxtyping import Bool
from torch import Tensor

from afabench.afa_rl.custom_types import AFAReward, AFARewardFn
from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def get_fixed_reward_reward_fn(
    reward_for_stop: float, reward_otherwise: float
) -> AFARewardFn:
    # Return
    def f(
        masked_features: MaskedFeatures,  # current masked features  # noqa: ARG001
        feature_mask: FeatureMask,  # current feature mask  # noqa: ARG001
        selection_mask: SelectionMask,  # noqa: ARG001
        new_masked_features: MaskedFeatures,  # new masked features  # noqa: ARG001
        new_feature_mask: FeatureMask,  # new feature mask  # noqa: ARG001
        new_selection_mask: SelectionMask,  # noqa: ARG001
        selection: AFASelection,  # noqa: ARG001
        features: Features,  # noqa: ARG001
        label: Label,  # noqa: ARG001
        done: Bool[Tensor, "*batch 1"],  # done key
    ) -> AFAReward:
        reward = reward_otherwise * torch.ones_like(done, dtype=torch.float32)
        done_mask = done.squeeze(-1)
        reward[done_mask] = reward_for_stop
        # Ensure reward maintains the same shape as done
        return reward.view_as(done).to(torch.float32)

    return f


def get_range_based_reward_fn(
    reward_ranges: list[tuple[int, int]], reward_value: float = 1.0
) -> AFARewardFn:
    """
    Return a reward function that gives a fixed reward for selections within specified ranges.

    Args:
        reward_ranges: List of (start, end) tuples indicating which selection indices give rewards
        reward_value: Reward value to give for selections in the ranges

    Returns:
        AFARewardFn that gives reward_value for selections in ranges, 0.0 otherwise
    """

    def f(
        masked_features: MaskedFeatures,  # noqa: ARG001
        feature_mask: FeatureMask,  # noqa: ARG001
        selection_mask: SelectionMask,
        new_masked_features: MaskedFeatures,  # noqa: ARG001
        new_feature_mask: FeatureMask,  # noqa: ARG001
        new_selection_mask: SelectionMask,
        afa_selection: AFASelection,  # noqa: ARG001
        features: Features,  # noqa: ARG001
        label: Label,  # noqa: ARG001
        done: Bool[Tensor, "*batch 1"],  # noqa: ARG001
    ) -> AFAReward:
        # Calculate newly performed selections
        newly_performed_selections = (new_selection_mask & ~selection_mask).to(
            torch.float32
        )

        # Initialize reward as zeros
        reward = torch.zeros_like(newly_performed_selections[..., 0:1])

        # For each range, add reward for selections in that range
        for start_idx, end_idx in reward_ranges:
            if end_idx < newly_performed_selections.shape[-1]:
                range_selections = newly_performed_selections[
                    ..., start_idx : end_idx + 1
                ]
                reward += (
                    range_selections.sum(dim=-1, keepdim=True) * reward_value
                )

        return reward

    return f
