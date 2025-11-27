from pathlib import Path
from typing import Self, final, override

import timm
import torch
from torch import nn

from afabench.common.custom_types import (
    AFAClassifier,
    FeatureMask,
    Features,
    Label,
    Logits,
    MaskedFeatures,
)
from afabench.common.models import MaskedMLPClassifier, MaskedViTClassifier


@final
class RandomDummyAFAClassifier(AFAClassifier):
    """A random dummy classifier that outputs random logits. It is used for testing purposes."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None,
        label: Label | None,
    ) -> Logits:
        # Return random logits with the same batch size as masked_features
        batch_size = masked_features.shape[0]
        logits = torch.randn(batch_size, self.n_classes)

        return logits

    @override
    def save(self, path: Path) -> None:
        """Saves the classifier to a file. n_classes is all we need."""
        torch.save(self.n_classes, path)

    @classmethod
    @override
    def load(
        cls, path: Path, device: torch.device
    ) -> "RandomDummyAFAClassifier":
        """Loads the classifier from a file, placing it on the given device."""
        # Load the number of classes
        n_classes = torch.load(path, map_location=device)

        # Return a new DummyClassifier instance
        return RandomDummyAFAClassifier(n_classes)


class UniformDummyAFAClassifier(AFAClassifier):
    """A uniform dummy classifier that outputs uniform logits. It is used for testing purposes."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        # Return random logits with the same batch size as masked_features
        batch_size = masked_features.shape[0]
        logits = torch.ones(batch_size, self.n_classes)

        return logits

    def save(self, path: Path) -> None:
        """Saves the classifier to a file. n_classes is all we need."""
        torch.save(self.n_classes, path)

    @staticmethod
    def load(path: str, device: torch.device) -> "UniformDummyAFAClassifier":
        """Loads the classifier from a file, placing it on the given device."""
        # Load the number of classes
        n_classes = torch.load(path, map_location=device)

        # Return a new DummyClassifier instance
        return UniformDummyAFAClassifier(n_classes)


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.model(x)


class NNClassifier(AFAClassifier):
    """
    A trainable classifier that uses a simple predictor
    and handles masked input.
    """

    def __init__(self, input_dim: int, output_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.predictor = Predictor(input_dim, output_dim).to(device)

    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Logits:
        x_masked = torch.cat([masked_features, feature_mask], dim=1)
        return self.predictor(x_masked)

    def save(self, path: Path) -> None:
        torch.save(
            {
                "model_state_dict": self.predictor.state_dict(),
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "NNClassifier":
        checkpoint = torch.load(path, map_location=device)
        classifier = cls(
            checkpoint["input_dim"], checkpoint["output_dim"], device
        )
        classifier.predictor.load_state_dict(checkpoint["model_state_dict"])
        return classifier


@final
class WrappedMaskedMLPClassifier(AFAClassifier):
    def __init__(self, module: MaskedMLPClassifier, device: torch.device):
        self.module = module.to(device)
        self.module.eval()
        self._device = device

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None = None,
        label: Label | None = None,
    ) -> Label:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        with torch.no_grad():
            logits = self.module(masked_features, feature_mask)
        return logits.softmax(dim=-1).to(original_device)

    @override
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.module.state_dict(),
                "n_features": self.module.n_features,
                "n_classes": self.module.n_classes,
                "num_cells": self.module.num_cells,
                "dropout": self.module.dropout,
            },
            path,
        )

    @override
    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        module = MaskedMLPClassifier(
            n_features=checkpoint["n_features"],
            n_classes=checkpoint["n_classes"],
            num_cells=tuple(checkpoint["num_cells"]),
            dropout=checkpoint["dropout"],
        )
        module.load_state_dict(checkpoint["state_dict"])
        module.eval()
        return cls(module=module, device=device)

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        self.module = self.module.to(device)
        return self


@final
class WrappedMaskedViTClassifier(AFAClassifier):
    def __init__(
        self,
        module: MaskedViTClassifier,
        device: torch.device,
        pretrained_model_name: str,
        image_size: int,
        patch_size: int,
    ):
        self.module = module.to(device)
        self.module.eval()
        self._device = device

        # Minimal reconstruction info
        self.pretrained_model_name = pretrained_model_name
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None,
        label: Label | None,
    ) -> Label:
        original_device = masked_features.device
        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        with torch.no_grad():
            logits = self.module(masked_features, feature_mask)
            probs = logits.softmax(dim=-1)

        return probs.to(original_device)

    @override
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.module.state_dict(),
            "num_classes": int(self.module.fc.out_features),
            "pretrained_model_name": self.pretrained_model_name,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
        }
        torch.save(checkpoint, path)

    @override
    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        name = checkpoint["pretrained_model_name"]
        num_classes = int(checkpoint["num_classes"])
        image_size = int(checkpoint["image_size"])
        patch_size = int(checkpoint["patch_size"])

        backbone = timm.create_model(name, pretrained=False)
        module = MaskedViTClassifier(
            backbone=backbone, num_classes=num_classes
        )
        module.load_state_dict(checkpoint["state_dict"])
        module.eval()

        return cls(
            module=module,
            device=device,
            pretrained_model_name=name,
            image_size=image_size,
            patch_size=patch_size,
        )

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        self.module = self.module.to(device)
        return self
