"""GDFS, greedy dynamic feature selection (Covert et al., ICML 2023)."""

from typing import ClassVar, override

import torch

from afabench.components.methods.discriminative.common.afa_methods import (
    GreedyDynamicSelection,
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


class GDFSAFAMethod(NetworkPairAFAMethod):
    """
    Acts on the selector's logits, and stops when the prediction is certain.

    The selector was trained through a Concrete relaxation to pick the
    selection that most improves the prediction; at evaluation its logit,
    per unit of selection cost, ranks the selections. The stop test is the
    predictor's entropy against `lambda_threshold`.
    """

    policy_network_name: ClassVar[str] = "selector"

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
        with torch.no_grad():
            pred = self.predictor(network_input)
            stop_mask = get_entropy(pred) < self.lambda_threshold
        logits = self.policy_network(network_input).flatten(1)
        scores = self._exclude_performed_selections(
            logits, feature_mask, selection_mask
        )
        return self._actions(self._score_per_cost(scores), stop_mask)


__all__ = ["GDFSAFAMethod", "GreedyDynamicSelection"]
