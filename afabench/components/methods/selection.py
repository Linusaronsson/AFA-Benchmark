from pathlib import Path
from typing import Self, final, override

import torch

from afabench.core.registry import get_class
from afabench.core.types import (
    AFAAction,
    AFAClassifier,
    AFAMethod,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


@final
class RandomSelectionAFAMethod(AFAMethod):
    """Select random unobserved features and delegate predictions to a classifier."""

    def __init__(
        self, afa_classifier: AFAClassifier, device: torch.device | None = None
    ):
        if device is None:
            self._device = afa_classifier.device
            self.afa_classifier = afa_classifier
        else:
            self._device = device
            self.afa_classifier = afa_classifier.to(device)

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        del selection_mask, label, feature_shape
        original_device = masked_features.device
        feature_mask = feature_mask.to(self._device)

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
        return selection.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        original_device = masked_features.device
        if label is not None:
            label = label.to(self._device)

        return self.afa_classifier(
            masked_features.to(self._device),
            feature_mask.to(self._device),
            label,
            feature_shape,
        ).to(original_device)

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.afa_classifier.save(path / "classifier.pt")
        with (path / "classifier_class_name.txt").open("w") as f:
            f.write(self.afa_classifier.__class__.__name__)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        with (path / "classifier_class_name.txt").open() as f:
            classifier_class_name = f.read()

        afa_classifier = get_class(classifier_class_name).load(
            path / "classifier.pt", device
        )
        return cls(afa_classifier, device)

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        self.afa_classifier.to(self._device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device


@final
class SequentialSelectionAFAMethod(AFAMethod):
    """Select the first unobserved feature and delegate predictions to a classifier."""

    def __init__(
        self, afa_classifier: AFAClassifier, device: torch.device | None = None
    ):
        if device is None:
            self._device = afa_classifier.device
            self.afa_classifier = afa_classifier
        else:
            self._device = device
            self.afa_classifier = afa_classifier.to(device)

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        del masked_features, selection_mask, label, feature_shape
        original_device = feature_mask.device
        feature_mask = feature_mask.to(self._device)

        selection = torch.zeros(
            (feature_mask.shape[0], 1), dtype=torch.long, device=self._device
        )
        for i in range(feature_mask.shape[0]):
            unobserved = (~feature_mask[i]).nonzero(as_tuple=True)[0]
            if unobserved.numel() > 0:
                selection[i, 0] = unobserved[0] + 1
        return selection.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        original_device = masked_features.device
        if label is not None:
            label = label.to(self._device)

        return self.afa_classifier(
            masked_features.to(self._device),
            feature_mask.to(self._device),
            label,
            feature_shape,
        ).to(original_device)

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.afa_classifier.save(path / "classifier.pt")
        with (path / "classifier_class_name.txt").open("w") as f:
            f.write(self.afa_classifier.__class__.__name__)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        with (path / "classifier_class_name.txt").open() as f:
            classifier_class_name = f.read()

        afa_classifier = get_class(classifier_class_name).load(
            path / "classifier.pt", device
        )
        return cls(afa_classifier, device)

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        self.afa_classifier.to(self._device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device
