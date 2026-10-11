"""DIME, discriminative mutual information estimation (Gadgil, Covert and Lee, 2023)."""

from typing import ClassVar, override

import torch

from afabench.components.methods.discriminative.common.afa_methods import (
    CMIEstimator,
    NetworkPairAFAMethod,
)
from afabench.components.methods.discriminative.common.utils import (
    get_entropy,
)
from afabench.core.types import (
    AFAAction,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


class DIMEAFAMethod(NetworkPairAFAMethod):
    """
    Acts on the value network's CMI estimates, and stops when none is worth its cost.

    The value network estimates each selection's conditional mutual
    information with the label, bounded by the predictor's entropy; at
    evaluation the estimate per unit of selection cost ranks the
    selections, and falls below `lambda_threshold` to stop.
    """

    policy_network_name: ClassVar[str] = "value_network"

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        network_input = self._network_input(masked_features, feature_mask)
        pred = self.predictor(network_input)
        entropy = get_entropy(pred).unsqueeze(1)
        pred_cmi = self.policy_network(network_input).sigmoid() * entropy
        scores = self._score_per_cost(
            self._exclude_performed_selections(
                pred_cmi, feature_mask, selection_mask
            )
        )
        stop_mask = scores.max(dim=1).values < self.lambda_threshold
        return self._actions(scores, stop_mask)


__all__ = ["CMIEstimator", "DIMEAFAMethod"]
