"""
L2M in-context acquisition, independently ported from arXiv:2510.12624.

AFABench never queries a context instance and uses the whole train split
as the pretraining pool, rather than the paper's 5000-instance pool.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Self, override

import torch

from afabench.components.methods.discriminative.l2m.models import L2MModel
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.types import (
    AFAAction,
    AFAMethod,
    ClassifierOutputKind,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
    SelectionMask,
)


class L2MAFAMethod(AFAMethod):
    """
    Fixed context set and a direct-selection policy with no stop action.

    An in-context method after Kobayashi et al., arXiv:2510.12624, ported
    independently of the authors' code. Departures from the paper: a
    context instance is never queried, because the context set comes from
    the validation split and queries from the test split, and the
    pretraining pool is the whole train split rather than 5000 instances.

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
        if unmasker.class_name != "DirectUnmasker":
            msg = f"L2M requires DirectUnmasker; got {unmasker.class_name!r}"
            raise ValueError(msg)
        if (
            context_features.ndim != 2
            or context_features.shape[0] == 0
            or context_features.shape[1] != model.n_features
        ):
            msg = (
                f"context_features shape {context_features.shape} must be "
                f"a nonempty matrix with {model.n_features} features"
            )
            raise ValueError(msg)
        expected_labels = (context_features.shape[0], model.n_classes)
        if context_labels.shape != expected_labels:
            msg = (
                f"context_labels shape {context_labels.shape} must be "
                f"{expected_labels}"
            )
            raise ValueError(msg)
        self.model: L2MModel = model
        self.context_features: Features = (
            context_features.detach().clone().to(model.device)
        )
        self.context_labels: Label = (
            context_labels.detach().clone().to(model.device)
        )
        self.unmasker: UnmaskerConfig = unmasker
        self.model.eval()

    @property
    @override
    def device(self) -> torch.device:
        return self.model.device

    @override
    def to(self, device: torch.device) -> Self:
        self.model.to(device)
        self.context_features = self.context_features.to(device)
        self.context_labels = self.context_labels.to(device)
        return self

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

    def _query_logits(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            masked_features.ndim != 2
            or masked_features.shape[1] != self.model.n_features
        ):
            msg = (
                f"masked_features shape {masked_features.shape} must be "
                f"a matrix with {self.model.n_features} features"
            )
            raise ValueError(msg)
        if feature_mask.shape != masked_features.shape:
            msg = (
                f"feature_mask shape {feature_mask.shape} must match "
                f"masked_features shape {masked_features.shape}"
            )
            raise ValueError(msg)
        features = torch.cat((self.context_features, masked_features))
        mask = torch.cat(
            (
                torch.ones_like(self.context_features, dtype=torch.bool),
                feature_mask,
            )
        )
        labels = torch.cat(
            (
                self.context_labels,
                masked_features.new_zeros(
                    (masked_features.shape[0], self.model.n_classes)
                ),
            )
        )
        return self.model(
            features, mask, labels, n_context=len(self.context_features)
        )

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @property
    def output_kind(self) -> ClassifierOutputKind:
        return "logits"

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self._query_logits(masked_features, feature_mask)
        return logits

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        with torch.no_grad():
            _, logits = self._query_logits(masked_features, feature_mask)
            performed = (
                feature_mask if selection_mask is None else selection_mask
            )
            if performed.shape != logits.shape:
                msg = (
                    f"selection_mask shape {performed.shape} does not "
                    f"match policy logits shape {logits.shape}"
                )
                raise ValueError(msg)
            # Evaluation removes stopped rows, not exhausted rows. Require
            # a sub-full hard budget instead of repeating a selection.
            if performed.all(dim=-1).any():
                msg = (
                    "L2M selections are exhausted for at least one query; "
                    "use a hard budget below the number of selections"
                )
                raise ValueError(msg)
            return (
                logits.masked_fill(performed, -torch.inf).argmax(
                    dim=-1, keepdim=True
                )
                + 1
            )
