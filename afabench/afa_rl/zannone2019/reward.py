import torch
from jaxtyping import Bool
from torch import Tensor

from afabench.afa_rl.common.custom_types import (
    AFAReward,
    AFARewardFn,
)
from afabench.afa_rl.zannone2019.models import Zannone2019PretrainingModel
from afabench.common.custom_types import (
    AFAAction,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def get_zannone2019_reward_fn(
    pretrained_model: Zannone2019PretrainingModel,
    weights: Tensor,  # noqa: ARG001
    selection_costs: torch.Tensor,
    n_feature_dims: int,
) -> AFARewardFn:
    """Return the reward function for zannone2019."""

    def f(
        masked_features: MaskedFeatures,  # noqa: ARG001
        _feature_mask: FeatureMask,
        _selection_mask: SelectionMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        new_selection_mask: SelectionMask,
        _afa_action: AFAAction,
        _features: Features,
        label: Label,
        _done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        # PVAE expects 1D features
        flat_new_masked_features = new_masked_features.flatten(
            start_dim=-n_feature_dims
        )
        flat_new_feature_mask = new_feature_mask.flatten(
            start_dim=-n_feature_dims
        )
        assert flat_new_masked_features.ndim == 2
        batch_size, _n_features = flat_new_masked_features.shape

        # We don't get to observe the label
        new_augmented_masked_features = torch.cat(
            [flat_new_masked_features, torch.zeros_like(label)], dim=-1
        )
        new_augmented_feature_mask = torch.cat(
            [flat_new_feature_mask, torch.full_like(label, False)], dim=-1
        )
        _encoding, mu, _logvar, _z = pretrained_model.partial_vae.encode(
            new_augmented_masked_features, new_augmented_feature_mask
        )
        logits = pretrained_model.classifier(mu)

        # Only consider prediction value of true class
        prob_true_class = logits.softmax(dim=-1)[
            torch.arange(batch_size), label.argmax(dim=-1)
        ]
        cost = (new_selection_mask.to(torch.float32) * selection_costs).sum(
            dim=-1
        )
        reward = prob_true_class - cost

        return reward

    return f
