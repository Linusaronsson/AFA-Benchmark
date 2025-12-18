import torch
import torch.nn.functional as F
from jaxtyping import Bool
from torch import Tensor

from afabench.afa_rl.common.custom_types import (
    AFAReward,
    AFARewardFn,
)
from afabench.afa_rl.shim2018.models import LitShim2018EmbedderClassifier
from afabench.common.custom_types import (
    AFASelection,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def get_shim2018_reward_fn(
    pretrained_model: LitShim2018EmbedderClassifier,
    weights: Tensor,
    acquisition_costs: torch.Tensor,
) -> AFARewardFn:
    """
    Return the reward function for shim2018.

    The agent receives the negative classification loss as reward at the end of the episode, and also a fixed
    negative reward for each selection made, encouraging it to select fewer features.
    """

    def f(
        _masked_features: MaskedFeatures,
        _feature_mask: FeatureMask,
        selection_mask: SelectionMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        new_selection_mask: SelectionMask,
        _afa_selection: AFASelection,
        features: Features,  # noqa: ARG001
        label: Label,
        done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        # Acquisition cost per selection
        newly_performed_selections = (new_selection_mask & ~selection_mask).to(
            torch.float32
        )
        reward = -(newly_performed_selections * acquisition_costs).sum(dim=-1)

        done_mask = done.squeeze(-1)

        if done_mask.any():
            _, logits = pretrained_model(
                new_masked_features[done_mask], new_feature_mask[done_mask]
            )
            assert logits.ndim == 2, (
                f"Expected logits to have 1 batch dimension and 1 label dimension, got {logits.ndim}"
            )
            # Reward=1 if correct prediction, 0 if wrong
            # correct = logits.argmax(dim=-1) == label[done_mask].argmax(dim=-1)
            # reward[done_mask] += correct.to(torch.float32)
            ce_loss = F.cross_entropy(
                logits,
                label[done_mask].float(),
                weight=weights,
                reduction="none",
            )
            reward[done_mask] += -ce_loss
            # print(f"Previous selection mask: {selection_mask[done_mask][0]}")
            # print(f"Made selection: {_afa_selection[done_mask][0]}")
            # print(f"ce loss: {ce_loss[0]}")

        return reward.unsqueeze(-1)

    return f
