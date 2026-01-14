import torch
from jaxtyping import Bool
from torch import Tensor
from torch.nn import functional as F

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
    weights: Tensor,
    selection_costs: torch.Tensor,
    n_feature_dims: int,
) -> AFARewardFn:
    """Return the reward function for zannone2019."""

    def f(
        _masked_features: MaskedFeatures,
        _feature_mask: FeatureMask,
        selection_mask: SelectionMask,
        new_masked_features: MaskedFeatures,
        new_feature_mask: FeatureMask,
        new_selection_mask: SelectionMask,
        _afa_action: AFAAction,
        _features: Features,
        label: Label,
        _done: Bool[Tensor, "*batch 1"],
    ) -> AFAReward:
        # Acquisition cost per selection
        newly_performed_selections = (new_selection_mask & ~selection_mask).to(
            torch.float32
        )
        reward = -(newly_performed_selections * selection_costs).sum(dim=-1)

        # PVAE expects 1D features
        flat_new_masked_features = new_masked_features.flatten(
            start_dim=-n_feature_dims
        )
        flat_new_feature_mask = new_feature_mask.flatten(
            start_dim=-n_feature_dims
        )

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
        ce_loss = F.cross_entropy(
            logits,
            label.float(),
            weight=weights,
            reduction="none",
        )
        reward += -ce_loss

        return reward

    return f
