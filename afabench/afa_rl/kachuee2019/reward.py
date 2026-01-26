import torch
from jaxtyping import Bool
from torch import Tensor

from afabench.afa_rl.common.custom_types import (
    AFAReward,
    AFARewardFn,
)
from afabench.afa_rl.kachuee2019.models import Kachuee2019PQModule
from afabench.common.custom_types import (
    AFAAction,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def calc_reward(conf_a: Tensor, conf_b: Tensor, method: str) -> torch.Tensor:
    """
    Calculate the reward according to eq. (7) in "Opportunistic Learning: Budgeted Cost-Sensitive Learning from Data Streams".

    Args:
        conf_a (Tensor of shape (batch_size, n_classes)): confidence for feature vector without new feature acquired
        conf_b (Tensor of shape (batch_size, n_classes)): confidence for feature vector with new feature acquired
        method (str): Which method to use. One of ("softmax", "Bayesian-L1", "Bayesian-L2")

    Returns: a Tensor with shape (batch_size,)

    """
    if method == "softmax":
        reward = torch.abs(conf_a.max(dim=-1)[0] - conf_b.max(dim=-1)[0])
    elif method == "Bayesian-L1":
        reward = torch.abs(conf_a - conf_b).sum(dim=-1)
    elif method == "Bayesian-L2":
        reward = ((conf_a - conf_b) ** 2.0).sum(dim=-1)
    else:
        msg = "Method is not supported:"
        raise NotImplementedError(msg, method)
    return reward


def get_kachuee2019_reward_fn(
    pretrained_model: Kachuee2019PQModule,
    selection_costs: torch.Tensor,
    n_feature_dims: int,
    method: str,
    mcdrop_samples: int,
) -> AFARewardFn:
    """
    Return the reward function for kachuee2019.

    The agent receives a reward at each step of the episode, equal to the relative confidence change.
    """

    def f(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        _selection_mask: SelectionMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        _new_selection_mask: SelectionMask,
        afa_action: AFAAction,
        _features: Features,
        _label: Label,
        _done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        conf_a = pretrained_model.confidence(
            masked_features.flatten(start_dim=-n_feature_dims),
            mcdrop_samples=mcdrop_samples,
            feature_mask=feature_mask.flatten(start_dim=-n_feature_dims),
        )
        conf_b = pretrained_model.confidence(
            new_masked_features.flatten(start_dim=-n_feature_dims),
            mcdrop_samples=mcdrop_samples,
            feature_mask=new_feature_mask.flatten(start_dim=-n_feature_dims),
        )
        unscaled_reward = calc_reward(conf_a, conf_b, method=method)
        action_cost = selection_costs[afa_action - 1]  # 1-based actions
        reward = unscaled_reward / action_cost
        return reward

    return f
