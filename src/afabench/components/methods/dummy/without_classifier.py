from pathlib import Path
from typing import Self, final, override

import torch

from afabench.components.methods.common import AFAMethodBase, on_method_device
from afabench.core.types import (
    AFAAction,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


@final
class RandomWithoutClassifierAFAMethod(AFAMethodBase):
    """A simple baseline AFAMethod that makes random AFA selections and random predictions."""

    def __init__(
        self,
        n_classes: int,
        prob_select_0: float = 0.0,
        device: torch.device | None = None,
    ):
        """
        Initialize.

        Args:
            n_classes: The number of classes for prediction.
            prob_select_0: The probability of selecting 0 (stop) at each time step.
            device: Which device to use.
        """
        super().__init__(torch.device("cpu") if device is None else device)
        self.n_classes: int = n_classes
        self.prob_select_0: float = prob_select_0

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
        """
        Chooses a random AFA selection, avoiding already occupied selections.

        If no available selections remain, outputs 0 even if prob_select_0 would not have triggered.
        """
        assert selection_mask is not None, (
            "RandomWithoutClassifierAFAMethod requires selection_mask to be provided"
        )
        del feature_mask, label, feature_shape

        batch_size = masked_features.shape[0]
        selection = -1 * torch.ones(
            (batch_size, 1),
            dtype=torch.long,
            device=masked_features.device,
        )
        will_select_0 = (
            torch.rand(batch_size, device=masked_features.device)
            < self.prob_select_0
        )

        for i in range(batch_size):
            if will_select_0[i]:
                selection[i, 0] = 0
            else:
                available = (
                    (~selection_mask[i]).nonzero(as_tuple=False).flatten()
                )
                if available.numel() == 0:
                    selection[i, 0] = 0
                else:
                    idx = torch.multinomial(
                        torch.ones(
                            available.numel(), device=masked_features.device
                        ),
                        num_samples=1,
                    )
                    selection[i, 0] = available[idx] + 1

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
        """Output a random prediction."""
        del feature_mask, label, feature_shape
        batch_size = masked_features.shape[0]

        random_class_onehot = torch.randint(
            0,
            self.n_classes,
            (batch_size,),
            device=masked_features.device,
        )
        return torch.nn.functional.one_hot(
            random_class_onehot, num_classes=self.n_classes
        ).float()

    @override
    def save(self, path: Path) -> None:
        """Save the method to a folder."""
        torch.save(
            {
                "n_classes": self.n_classes,
                "prob_select_0": self.prob_select_0,
            },
            path / "method.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a file."""
        data = torch.load(path / "method.pt")
        return cls(
            n_classes=data["n_classes"],
            prob_select_0=data["prob_select_0"],
            device=device,
        )

    @property
    @override
    def cost_param(self) -> float:
        return self.prob_select_0


@final
class SequentialWithoutClassifierAFAMethod(AFAMethodBase):
    """A simple baseline AFAMethod that selects features in order and makes random predictions."""

    def __init__(
        self,
        device: torch.device,
        n_classes: int,
        prob_select_0: float,
    ):
        super().__init__(device)
        self.n_classes: int = n_classes
        self.prob_select_0: float = prob_select_0

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
        assert selection_mask is not None, (
            "SequentialWithoutClassifierAFAMethod requires selection_mask to be provided"
        )
        del feature_mask, label, feature_shape

        batch_size = masked_features.shape[0]
        select_0_mask = (
            torch.rand(batch_size, device=self._device) < self.prob_select_0
        )

        selection = -1 * torch.ones(
            (batch_size,), dtype=torch.long, device=self._device
        )
        for i in range(batch_size):
            unperformed = (~selection_mask[i]).nonzero(as_tuple=True)[0]
            if unperformed.numel() > 0:
                selection[i] = unperformed[0] + 1
            else:
                selection[i] = 0

        selection = torch.where(
            select_0_mask, torch.zeros_like(selection), selection
        )

        return selection.unsqueeze(-1)

    @override
    @on_method_device
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """Output a random prediction."""
        del feature_mask, label, feature_shape
        batch_size = masked_features.shape[0]

        random_class_onehot = torch.randint(
            0,
            self.n_classes,
            (batch_size,),
            device=masked_features.device,
        )
        return torch.nn.functional.one_hot(
            random_class_onehot, num_classes=self.n_classes
        ).float()

    @override
    def save(self, path: Path) -> None:
        """Save the method to a folder."""
        torch.save(
            {
                "n_classes": self.n_classes,
                "prob_select_0": self.prob_select_0,
            },
            path / "method.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a folder."""
        data = torch.load(path / "method.pt")
        return cls(
            device=device,
            n_classes=data["n_classes"],
            prob_select_0=data["prob_select_0"],
        )

    @property
    @override
    def cost_param(self) -> float:
        return self.prob_select_0
