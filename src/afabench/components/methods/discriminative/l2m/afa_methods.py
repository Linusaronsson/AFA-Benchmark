"""
L2M in-context acquisition with the authors' code (arXiv:2510.12624).

Acting and predicting follow upstream's evaluation (`test_afa` in
`src/utils/eval_utils.py` at the vendored commit), apart from the benchmark's
context set; see `L2MAFAMethod`.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Self, override

import torch

from afabench.components.methods.common.afa_method_base import (
    AFAMethodBase,
    on_method_device,
)
from afabench.components.methods.discriminative.l2m.models import (
    L2MModel,
    UpstreamBatch,
)
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.types import (
    AFAAction,
    ClassifierOutputKind,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def require_direct_unmasker(unmasker: UnmaskerConfig) -> None:
    if unmasker.class_name != "DirectUnmasker":
        msg = f"L2M requires DirectUnmasker; got {unmasker.class_name!r}"
        raise ValueError(msg)


# Upstream's `RealDataSampler._normalize` guard against a zero deviation.
_STD_EPSILON = 1e-8


class L2MAFAMethod(AFAMethodBase):
    """
    Fixed context set and the authors' policy with no stop action.

    As upstream's evaluation, the query instances' features are normalized
    with the context set's feature means and deviations, query instances
    read only the context set and themselves, and the policy takes the
    argmax over features not yet selected. Unlike upstream, whose context
    set is a random draw from its test pool with simulated missingness, the
    context set is the benchmark's: a fixed draw from the validation split,
    fully observed.

    Evaluation must use a hard budget below the number of selections.
    Exhausted selections raise rather than invent a stop or repeat action.
    """

    def __init__(
        self,
        model: L2MModel,
        context_features: Features,
        context_labels: Label,
        *,
        unmasker: UnmaskerConfig,
    ) -> None:
        require_direct_unmasker(unmasker)
        if (
            context_features.ndim != 2
            or context_features.shape[0] < 2
            or context_features.shape[1] != model.n_features
        ):
            msg = (
                f"context_features shape {context_features.shape} must be "
                f"a matrix of at least two instances with "
                f"{model.n_features} features"
            )
            raise ValueError(msg)
        expected_labels = (context_features.shape[0], model.n_classes)
        if context_labels.shape != expected_labels:
            msg = (
                f"context_labels shape {context_labels.shape} must be "
                f"{expected_labels}"
            )
            raise ValueError(msg)
        super().__init__(model.device)
        self.model: L2MModel = model.eval()
        self.context_features: Features = (
            context_features.detach().clone().to(model.device)
        )
        self.context_labels: Label = (
            context_labels.detach().clone().to(model.device)
        )
        self.unmasker: UnmaskerConfig = unmasker

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model")
        torch.save(
            {
                "context_features": self.context_features,
                "context_labels": self.context_labels,
                "unmasker": asdict(self.unmasker),
            },
            path / "method.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(
            path / "method.pt", map_location=device, weights_only=True
        )
        return cls(
            L2MModel.load(path / "model", device),
            checkpoint["context_features"],
            checkpoint["context_labels"],
            unmasker=UnmaskerConfig(**checkpoint["unmasker"]),
        )

    def _batch(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> UpstreamBatch:
        """One task in upstream's layout: the context set, then the queries."""
        n_features = self.model.n_features
        if masked_features.ndim != 2 or masked_features.shape[1] != n_features:
            msg = (
                f"masked_features shape {masked_features.shape} must be "
                f"a matrix with {n_features} features"
            )
            raise ValueError(msg)
        if feature_mask.shape != masked_features.shape:
            msg = (
                f"feature_mask shape {feature_mask.shape} must match "
                f"masked_features shape {masked_features.shape}"
            )
            raise ValueError(msg)
        mean = self.context_features.mean(dim=0)
        std = self.context_features.std(dim=0) + _STD_EPSILON
        query_mask = feature_mask.to(masked_features.dtype)
        context_labels = self.model.upstream_labels(self.context_labels)
        return {
            "xc": ((self.context_features - mean) / std).unsqueeze(0),
            "yc": context_labels.unsqueeze(0),
            "rc": torch.ones_like(self.context_features).unsqueeze(0),
            "xt": ((masked_features - mean) / std).unsqueeze(0),
            "yt": context_labels.new_zeros(
                (1, len(masked_features), context_labels.shape[-1])
            ),
            # Every query feature can be acquired; the mask is what has
            # been.
            "rt": torch.ones_like(query_mask).unsqueeze(0),
            "maskt": query_mask.unsqueeze(0),
        }

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @property
    @override
    def output_kind(self) -> ClassifierOutputKind:
        return "logits"

    @override
    @on_method_device
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self.model(self._batch(masked_features, feature_mask))[0]

    @override
    @on_method_device
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        with torch.no_grad():
            logits = self.model.ltm.select_action(
                self._batch(masked_features, feature_mask)
            )[0]
        performed = feature_mask if selection_mask is None else selection_mask
        if performed.shape != logits.shape:
            msg = (
                f"selection_mask shape {performed.shape} does not "
                f"match policy logits shape {logits.shape}"
            )
            raise ValueError(msg)
        # Evaluation removes stopped instances, not exhausted ones. Require a
        # sub-full hard budget instead of repeating a selection.
        if performed.bool().all(dim=-1).any():
            msg = (
                "L2M selections are exhausted for at least one query "
                "instance; use a hard budget below the number of selections"
            )
            raise ValueError(msg)
        return (
            logits.masked_fill(performed.bool(), -torch.inf).argmax(
                dim=-1, keepdim=True
            )
            + 1
        )
