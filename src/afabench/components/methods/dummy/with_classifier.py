from pathlib import Path
from typing import Self, final, override

import torch

from afabench.components.methods.common import (
    AFAMethodBase,
    load_classifier,
    on_method_device,
    save_classifier,
)
from afabench.core.types import (
    AFAAction,
    AFAClassifier,
    ClassifierOutputKind,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


@final
class RandomWithClassifierAFAMethod(AFAMethodBase):
    """Select random unobserved features and delegate predictions to a classifier."""

    def __init__(
        self, afa_classifier: AFAClassifier, device: torch.device | None = None
    ):
        super().__init__(afa_classifier.device if device is None else device)
        self.afa_classifier: AFAClassifier = afa_classifier.to(self._device)

    @property
    @override
    def output_kind(self) -> ClassifierOutputKind:
        return self.afa_classifier.output_kind

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

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
        del masked_features, selection_mask, label, feature_shape

        probs = (~feature_mask).float()
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = torch.where(row_sums > 0, probs / row_sums, probs)

        selection = torch.zeros(
            (feature_mask.shape[0], 1), dtype=torch.long, device=self._device
        )
        can_select = row_sums.squeeze(-1) > 0
        if can_select.any():
            selection[can_select] = (
                torch.multinomial(probs[can_select], num_samples=1) + 1
            )
        return selection

    @override
    @on_method_device
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        return self.afa_classifier(
            masked_features, feature_mask, label, feature_shape
        )

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_classifier(self.afa_classifier, path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        return cls(load_classifier(path, device), device)


@final
class SequentialWithClassifierAFAMethod(AFAMethodBase):
    """Select the first unobserved feature and delegate predictions to a classifier."""

    def __init__(
        self, afa_classifier: AFAClassifier, device: torch.device | None = None
    ):
        super().__init__(afa_classifier.device if device is None else device)
        self.afa_classifier: AFAClassifier = afa_classifier.to(self._device)

    @property
    @override
    def output_kind(self) -> ClassifierOutputKind:
        return self.afa_classifier.output_kind

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

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
        del masked_features, selection_mask, label, feature_shape

        selection = torch.zeros(
            (feature_mask.shape[0], 1), dtype=torch.long, device=self._device
        )
        for i in range(feature_mask.shape[0]):
            unobserved = (~feature_mask[i]).nonzero(as_tuple=True)[0]
            if unobserved.numel() > 0:
                selection[i, 0] = unobserved[0] + 1
        return selection

    @override
    @on_method_device
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        return self.afa_classifier(
            masked_features, feature_mask, label, feature_shape
        )

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        save_classifier(self.afa_classifier, path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        return cls(load_classifier(path, device), device)
