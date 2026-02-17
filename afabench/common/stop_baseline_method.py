import torch

from pathlib import Path
from typing import Self, final, override

from afabench.common.registry import get_class
from afabench.common.custom_types import (
    AFAAction,
    AFAClassifier,
    AFAMethod,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


@final
class StopBaselineMethod(AFAMethod):
    """
    Stop baseline AFAMethod (pi ≡ STOP).

    This serves as the pi ≡ STOP baseline: predictions are made using only
    the initially observed features. When paired with a MissingnessInitializer,
    this emulates the behavior of MA-learn methods (MA-DT, MA-RF, etc.) that
    predict directly on partially observed data without feature acquisition.

    In the framework of Stempfle et al., this corresponds to a policy that
    always stops immediately, relying on a classifier trained to handle
    partial observations (e.g., MA-DT, MA-LASSO, MA-RF, MA-GBT).

    The classifier should be trained to handle masked/partial inputs (e.g.,
    a WrappedMaskedMLPClassifier trained with random masking).
    """

    def __init__(
        self,
        afa_classifier: AFAClassifier,
        device: torch.device | None = None,
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
        """Always return 0 (stop action) immediately."""
        batch_size = masked_features.shape[0]
        return torch.zeros(
            (batch_size, 1), dtype=torch.long, device=masked_features.device
        )

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """Predict using only the currently observed features."""
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
        with (path / "classifier_class_name.txt").open("r") as f:
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
