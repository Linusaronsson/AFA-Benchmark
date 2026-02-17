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
    AFAAction,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def get_shim2018_reward_fn(
    pretrained_model: LitShim2018EmbedderClassifier,
    weights: Tensor,
    selection_costs: torch.Tensor,
    n_feature_dims: int,
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
        _afa_action: AFAAction,
        features: Features,  # noqa: ARG001
        label: Label,
        done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        # Acquisition cost per selection
        newly_performed_selections = (new_selection_mask & ~selection_mask).to(
            torch.float32
        )
        reward = -(newly_performed_selections * selection_costs).sum(dim=-1)

        done_mask = done.squeeze(-1)

        if done_mask.any():
            _, logits = pretrained_model(
                new_masked_features[done_mask].flatten(
                    start_dim=-n_feature_dims
                ),
                new_feature_mask[done_mask].flatten(start_dim=-n_feature_dims),
            )
            assert (
                logits.ndim == 2
            ), f"Expected logits to have 1 batch dimension and 1 label dimension, got {logits.ndim}"
            ce_loss = F.cross_entropy(
                logits,
                label[done_mask].float(),
                weight=weights,
                reduction="none",
            )
            reward[done_mask] += -ce_loss

        return reward.unsqueeze(-1)

    return f
