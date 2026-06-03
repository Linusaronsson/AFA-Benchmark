from pathlib import Path
from typing import Self, final, override

import torch

from afabench.core.types import (
    AFAAction,
    AFAMethod,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


@final
class RandomDummyAFAMethod(AFAMethod):
    """A dummy AFAMethod for testing purposes. Makes random AFA selections."""

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
        self.n_classes = n_classes
        self.prob_select_0 = prob_select_0
        if device is None:
            self._device = torch.device("cpu")
        else:
            self._device = device

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
        """
        Chooses a random AFA selection, avoiding already occupied selections.

        If no available selections remain, outputs 0 even if prob_select_0 would not have triggered.
        """
        assert selection_mask is not None, (
            "RandomDummyAFAMethod requires selection_mask to be provided"
        )
        # RandomDummyAFAMethod works with any feature shape since it only uses selection_mask
        original_device = masked_features.device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)
        selection_mask = selection_mask.to(self._device)

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
                    # No available selections left, must output 0
                    selection[i, 0] = 0
                else:
                    # Pick one at random, add 1 for 1-based index
                    idx = torch.multinomial(
                        torch.ones(
                            available.numel(), device=masked_features.device
                        ),
                        num_samples=1,
                    )
                    selection[i, 0] = available[idx] + 1

        selection = selection.to(original_device)
        return selection

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """Output a random prediction."""
        original_device = masked_features.device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        batch_size = masked_features.shape[0]

        # Pick a random class from the classes
        random_class_onehot = torch.randint(
            0,
            self.n_classes,
            (batch_size,),
            device=masked_features.device,
        )
        # One-hot encode the random prediction
        random_class_onehot = torch.nn.functional.one_hot(
            random_class_onehot, num_classes=self.n_classes
        ).float()
        random_class_onehot = random_class_onehot.to(original_device)
        return random_class_onehot

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
        obj = cls.__new__(cls)
        obj.n_classes = data["n_classes"]
        obj.prob_select_0 = data["prob_select_0"]
        obj._device = device  # noqa: SLF001
        return obj

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def cost_param(self) -> float:
        # Probability of selecting 0 can be interpreted as a cost parameter
        return self.prob_select_0


@final
class SequentialDummyAFAMethod(AFAMethod):
    """A dummy AFAMethod for testing purposes. Always chooses the next feature to observe in order, with a probability to stop."""

    def __init__(
        self,
        device: torch.device,
        n_classes: int,
        prob_select_0: float,
    ):
        self._device = device
        self.n_classes = n_classes
        self.prob_select_0 = prob_select_0

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
        # Requires the selection mask to be given so that we don't
        # repeat selections and know which selection to perform next
        assert selection_mask is not None, (
            "SequentialDummyAFAMethod requires selection_mask to be provided"
        )
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        batch_size = masked_features.shape[0]
        select_0_mask = (
            torch.rand(batch_size, device=self._device) < self.prob_select_0
        )

        # For each sample, find the first unperformed selection (if any)
        # Use -1 to catch errors in case the logic is wrong
        selection = -1 * torch.ones(
            (batch_size,), dtype=torch.long, device=self._device
        )
        for i in range(batch_size):
            unperformed = (~selection_mask[i]).nonzero(as_tuple=True)[0]
            if unperformed.numel() > 0:
                selection[i] = unperformed[0] + 1
            else:
                selection[i] = 0  # fallback if all selections are performed

        # Where select_0_mask is True, set selection to 0
        selection = torch.where(
            select_0_mask, torch.zeros_like(selection), selection
        )

        return selection.to(original_device).unsqueeze(-1)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """Output a random prediction."""
        original_device = masked_features.device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        batch_size = masked_features.shape[0]

        # Pick a random class from the classes
        random_class_onehot = torch.randint(
            0,
            self.n_classes,
            (batch_size,),
            device=masked_features.device,
        )
        # One-hot encode the random prediction
        random_class_onehot = torch.nn.functional.one_hot(
            random_class_onehot, num_classes=self.n_classes
        ).float()
        random_class_onehot = random_class_onehot.to(original_device)
        return random_class_onehot

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
        obj = cls.__new__(cls)
        obj.n_classes = data["n_classes"]
        obj.prob_select_0 = data["prob_select_0"]
        obj._device = device  # noqa: SLF001
        return obj

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def cost_param(self) -> float:
        return self.prob_select_0
