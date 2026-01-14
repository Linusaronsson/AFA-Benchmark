from pathlib import Path
from typing import Self, final, override

import torch

from afabench.common.custom_types import (
    AFAAction,
    AFAClassifier,
    AFAMethod,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.common.registry import get_class


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


@final
class RandomClassificationAFAMethod(AFAMethod):
    """An AFAMethod that randomly selects new features but uses a trained classifier for predictions."""

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
        """Chooses to observe a random feature from the masked features (or stop collecting features)."""
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Sample from unobserved features uniformly
        probs = (~feature_mask).float()

        # Avoid division by zero
        row_sums = probs.sum(dim=1, keepdim=True)
        probs = torch.where(
            row_sums > 0, probs / row_sums, probs
        )  # normalize or leave zeros

        # Sample one index per row
        sampled = torch.multinomial(probs, num_samples=1)
        # selection = sampled.squeeze(1)  # (B, 1) → (B,)
        selection = sampled + 1

        return selection.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """Return a prediction using the classifier."""
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
        """Save the method to a folder."""
        self.afa_classifier.save(path / "classifier.pt")
        # Write the classifier class name to a file
        with (path / "classifier_class_name.txt").open("w") as f:
            f.write(self.afa_classifier.__class__.__name__)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a file."""
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


@final
class RandomPatchClassificationAFAMethod(AFAMethod):
    def __init__(
        self,
        afa_classifier: AFAClassifier,
        image_side_length: int,
        patch_size: int,
        device: torch.device | None = None,
    ):
        if device is None:
            self._device = afa_classifier.device
            self.afa_classifier = afa_classifier
        else:
            self._device = device
            self.afa_classifier = afa_classifier.to(device)

        assert image_side_length % patch_size == 0
        self.image_side_length = image_side_length
        self.patch_size = patch_size
        self.low_dim_side = image_side_length // patch_size
        self.n_patches = self.low_dim_side**2

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
        """Randomly select an unseen patch (1-based index in [1, n_patches])."""
        original_device = masked_features.device
        feature_mask = feature_mask.to(self._device)
        # 4D image
        assert feature_mask.dim() == 4
        B, C, H, W = feature_mask.shape
        ph = H // self.patch_size
        pw = W // self.patch_size
        assert ph == self.low_dim_side
        assert pw == self.low_dim_side
        fm = feature_mask.view(
            B,
            C,
            ph,
            self.patch_size,
            pw,
            self.patch_size,
        )
        patch_revealed = fm.any(dim=(1, 3, 5))
        patch_probs = (~patch_revealed).view(B, -1).float()
        row_sums = patch_probs.sum(dim=1, keepdim=True)
        assert torch.all(row_sums > 0)
        patch_probs = patch_probs / row_sums

        # 1-based index
        # TODO: check the index tabular random selection class and hard budget evaluation function
        selection = (
            torch.multinomial(patch_probs, num_samples=1).squeeze(1) + 1
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
        """Return a prediction using the classifier."""
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
        # Write the classifier class name to a file
        with (path / "classifier_class_name.txt").open("w") as f:
            f.write(self.afa_classifier.__class__.__name__)

        torch.save(
            {
                "image_side_length": self.image_side_length,
                "patch_size": self.patch_size,
            },
            path / "config.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a file."""
        with (path / "classifier_class_name.txt").open("r") as f:
            classifier_class_name = f.read()

        afa_classifier = get_class(classifier_class_name).load(
            path / "classifier.pt", device
        )
        cfg = torch.load(path / "config.pt")
        image_side_length = int(cfg["image_side_length"])
        patch_size = int(cfg["patch_size"])
        return cls(
            afa_classifier=afa_classifier,
            image_side_length=image_side_length,
            patch_size=patch_size,
            device=device,
        )

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
class SequentialClassificationAFAMethod(AFAMethod):
    """An AFAMethod that selects features sequentially and uses a trained classifier to predict labels."""

    def __init__(
        self, afa_classifier: AFAClassifier, device: torch.device | None = None
    ):
        if device is None:
            self._device = afa_classifier.device
            self.afa_classifier = afa_classifier
        else:
            self._device = device
            self.afa_classifier = afa_classifier.to(device)

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Choose the next unobserved feature
        unobserved_features = (~feature_mask).nonzero(as_tuple=True)[1]
        if unobserved_features.numel() == 0:
            return torch.tensor(0, device=masked_features.device)
        selection = unobserved_features[0] + 1

        return selection.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """Return a prediction using the classifier."""
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
        """Save the method to a folder."""
        self.afa_classifier.save(path / "classifier.pt")
        # Write the classifier class name to a file
        with (path / "classifier_class_name.txt").open("w") as f:
            f.write(self.afa_classifier.__class__.__name__)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a file."""
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


@final
class OptimalCubeAFAMethod(AFAMethod):
    """An AFAMethod that selects features optimally for the cube dataset. It does this by looking at the true label and choosing the 3 informative features. Afterwards, it chooses features randomly."""

    def __init__(self, device: torch.device, n_classes: int):
        self._device = device
        self.n_classes = n_classes

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        """Chooses to observe an optimal feature for the cube dataset, or a random unobserved feature if all optimal ones are chosen."""
        original_device = masked_features.device

        assert label is not None, (
            "OptimalCubeAFAMethod assumes that label is available"
        )

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)
        label = label.to(self._device)

        batch_size, _num_features = feature_mask.shape
        label_int = label.argmax(dim=-1)  # (B,)

        selection = torch.zeros(
            batch_size, dtype=torch.long, device=self._device
        )

        for i in range(batch_size):
            optimal_idxs = torch.arange(
                label_int[i].item(),
                label_int[i].item() + 3,
                device=feature_mask.device,
            )
            unobserved = (~feature_mask[i])[optimal_idxs]
            if unobserved.any():
                selection[i] = optimal_idxs[unobserved][0]
            else:
                # Pick a random unobserved feature
                candidates = (~feature_mask[i]).nonzero(as_tuple=True)[0]
                if candidates.numel() > 0:
                    selection[i] = candidates[
                        torch.randint(0, candidates.numel(), (1,))
                    ]
                else:
                    selection[i] = 0  # fallback if all features observed

        return selection.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """Return a random prediction from the classes."""
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Pick a random class from the classes
        prediction = torch.randint(
            0,
            self.n_classes,
            (masked_features.shape[0],),
            device=masked_features.device,
        )
        # One-hot encode the prediction
        prediction = torch.nn.functional.one_hot(
            prediction, num_classes=self.n_classes
        ).float()

        return prediction.to(original_device)

    @override
    def save(self, path: Path) -> None:
        """Save the method to a folder."""
        torch.save(
            {
                "n_classes": self.n_classes,
            },
            path / "method.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a file."""
        data = torch.load(path / "method.pt")
        return cls(device, data["n_classes"])

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device
