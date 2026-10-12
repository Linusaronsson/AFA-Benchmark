"""
The AFA method base GDFS and DIME share.

Both come from one upstream code base (Covert et al., ICML 2023; Gadgil,
Covert and Lee, 2023), so the port shares what that code base shared and
keeps each paper's own act and training step explicit in the method's
package. Their shared training loop is `trainer.py`; what the shared code
does differently from the upstream implementations is written up in
`docs/explanation/gdfs_dime_shared_code.md`.
"""

import math
from abc import ABC
from pathlib import Path
from typing import ClassVar, Self, override

import torch
from torch import Tensor, nn

from afabench.components.methods.common import AFAMethodBase
from afabench.components.methods.discriminative.common.checkpoint import (
    NetworkPairSpec,
)
from afabench.core.types import (
    AFAAction,
    ClassifierOutputKind,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def _append_flat_mask(
    masked_features: torch.Tensor, feature_mask: torch.Tensor
) -> torch.Tensor:
    if masked_features.dim() > 2:
        masked_features = masked_features.flatten(start_dim=1)
        feature_mask = feature_mask.flatten(start_dim=1)
    return torch.cat([masked_features, feature_mask], dim=1)


class NetworkPairAFAMethod(AFAMethodBase, ABC):
    """
    The method shape GDFS and DIME share: a policy network and a classifier.

    Both methods score every selection with a policy network, exclude the
    selections already performed, scale the scores by the selection costs
    and stop below a threshold; both predict with a built-in classifier,
    the `classifier`, that takes the masked features (with the feature mask
    appended for tabular data). The base owns that, the device moves and
    the model.pt round trip through `NetworkPairSpec`; a subclass names
    its policy network for the checkpoint and implements `act`.

    `lambda_threshold` is the soft-budget parameter; without one the method
    never stops on its own, so the hard-budget setting forces every stop.
    """

    policy_network_name: ClassVar[str]
    """The checkpoint's name for the policy network: `selector` or `value_network`."""

    def __init__(
        self,
        policy_network: nn.Module,
        classifier: nn.Module,
        spec: NetworkPairSpec,
        device: torch.device,
        *,
        lambda_threshold: float | None = None,
        selection_costs: Tensor | None = None,
    ) -> None:
        super().__init__(device)
        self.policy_network: nn.Module = policy_network
        self.classifier: nn.Module = classifier
        self.spec: NetworkPairSpec = spec
        self.lambda_threshold: float = (
            -math.inf if lambda_threshold is None else lambda_threshold
        )
        self.selection_costs: Tensor | None = selection_costs

    @property
    @override
    def output_kind(self) -> ClassifierOutputKind:
        return "probabilities"

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @property
    @override
    def cost_param(self) -> float | None:
        return float(self.lambda_threshold)

    @override
    def set_cost_param(self, cost_param: float) -> None:
        self.lambda_threshold = cost_param

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        pred = self.classifier(
            self._network_input(masked_features, feature_mask)
        )
        return pred.softmax(dim=-1)

    def _network_input(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Tensor:
        if self.spec.modality == "tabular":
            return _append_flat_mask(masked_features, feature_mask)
        return masked_features

    def _exclude_performed_selections(
        self,
        scores: Tensor,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None,
    ) -> Tensor:
        """Push the scores of the selections already performed below every other score."""
        if self.spec.modality == "image":
            return scores - 1e6 * self._patch_mask(feature_mask).float()
        if selection_mask is not None:
            if scores.shape != selection_mask.shape:
                msg = (
                    f"selection_mask shape {selection_mask.shape} is "
                    "incompatible with the policy network's output shape "
                    f"{scores.shape}."
                )
                raise ValueError(msg)
            return scores - 1e6 * selection_mask.float()
        if scores.shape == feature_mask.shape:
            # Selections map 1:1 onto features (the direct Unmasker), so
            # the feature mask doubles as the selection mask
            return scores - 1e6 * feature_mask
        msg = (
            f"{type(self).__name__} requires a selection_mask when "
            "selections do not correspond 1:1 to features (e.g. grouped "
            "selections such as the Context Unmasker on cube_nm): policy "
            f"network output shape {scores.shape} does not match "
            f"feature_mask shape {feature_mask.shape}."
        )
        raise ValueError(msg)

    def _patch_mask(self, feature_mask: FeatureMask) -> Tensor:
        """Whether any pixel of each patch is revealed, one column per patch."""
        assert feature_mask.dim() == 4
        patch_size = self.spec.patch_size
        assert patch_size is not None
        B, C, H, W = feature_mask.shape
        ph = H // patch_size
        pw = W // patch_size
        fm = feature_mask.view(B, C, ph, patch_size, pw, patch_size)
        return fm.any(dim=(1, 3, 5)).reshape(B, ph * pw)

    def _score_per_cost(self, scores: Tensor) -> Tensor:
        if self.selection_costs is None:
            return scores
        costs = torch.clamp(self.selection_costs.to(self.device), min=1e-12)
        return scores / costs.unsqueeze(0)

    def _actions(self, scores: Tensor, stop_mask: Tensor) -> AFAAction:
        """Encode the best-scoring selection as an action, or stop where `stop_mask` says so or every selection is excluded."""
        best_scores, best_idx = scores.max(dim=1)
        stop_mask = stop_mask | (best_scores < -1e5)
        selections = (best_idx + 1).to(dtype=torch.long).unsqueeze(-1)
        return selections.masked_fill(stop_mask.unsqueeze(-1), 0)

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            f"{self.policy_network_name}_state_dict": (
                self.policy_network.state_dict()
            ),
            "classifier_state_dict": self.classifier.state_dict(),
            "architecture": self.spec.to_dict(self.policy_network_name),
            "lambda_threshold": float(self.lambda_threshold),
            "selection_costs": self.selection_costs.detach().cpu()
            if self.selection_costs is not None
            else None,
        }
        torch.save(payload, path / "model.pt")

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(
            path / "model.pt", weights_only=False, map_location=device
        )
        spec = NetworkPairSpec.from_dict(
            checkpoint["architecture"], cls.policy_network_name
        )
        policy_network, classifier = spec.build_networks(
            device, share_image_backbone=cls.policy_network_name != "selector"
        )
        policy_network.load_state_dict(
            checkpoint[f"{cls.policy_network_name}_state_dict"]
        )
        classifier.load_state_dict(checkpoint["classifier_state_dict"])
        policy_network.eval()
        classifier.eval()
        selection_costs = checkpoint.get("selection_costs")
        if selection_costs is None:
            # Bundles from before selection costs stored the feature
            # costs, which equal them under the direct Unmasker
            selection_costs = checkpoint.get("feature_costs")
        if selection_costs is not None:
            selection_costs = selection_costs.to(device)
        return cls(
            policy_network,
            classifier,
            spec,
            device,
            lambda_threshold=checkpoint.get("lambda_threshold"),
            selection_costs=selection_costs,
        )
