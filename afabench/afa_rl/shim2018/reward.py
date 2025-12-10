import torch
from jaxtyping import Bool
from torch import Tensor

from afabench.afa_rl.custom_types import (
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
    _pretrained_model: LitShim2018EmbedderClassifier,
    _weights: Tensor,
    _acquisition_costs: torch.Tensor,
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
        _new_masked_features: MaskedFeatures,
        _new_feature_mask: FeatureMask,
        new_selection_mask: SelectionMask,
        _afa_selection: AFASelection,
        features: Features,  # noqa: ARG001
        _label: Label,
        _done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        # Acquisition cost per selection
        # newly_performed_selections = (new_selection_mask & ~selection_mask).to(
        #     torch.float32
        # )
        # reward = -(newly_performed_selections * acquisition_costs).sum(dim=-1)

        # done_mask = done.squeeze(-1)

        # if done_mask.any():
        #     _, logits = pretrained_model(
        #         new_masked_features[done_mask], new_feature_mask[done_mask]
        #     )
        #     reward[done_mask] = -F.cross_entropy(
        #         logits, label[done_mask], weight=weights
        #     )

        # return reward.unsqueeze(-1)
        # DEBUG START
        # Give a fixed reward every time selections 5-10 are made, just for debugging
        newly_performed_selections = (new_selection_mask & ~selection_mask).to(
            torch.float32
        )
        reward = newly_performed_selections[..., 5:11].sum(
            dim=-1, keepdim=True
        )
        return reward

        # DEBUG END

    return f
