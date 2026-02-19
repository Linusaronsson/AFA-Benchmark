from pathlib import Path
from typing import Any, Self, final, override

import numpy as np
import timm
import torch
from torch import Tensor, nn

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
        """Save the classifier to a file. Only n_classes needs to be stored."""
        torch.save(self.n_classes, path)

    @classmethod
    @override
    def load(
        cls, path: Path, device: torch.device
    ) -> "RandomDummyAFAClassifier":
        """Load the classifier from a file, placing it on the given device."""
        # Load the number of classes
        n_classes = torch.load(path, map_location=device)

        # Return a new DummyClassifier instance
        return RandomDummyAFAClassifier(n_classes)


class UniformDummyAFAClassifier(AFAClassifier):
    """A uniform dummy classifier that outputs uniform logits. It is used for testing purposes."""

    def __init__(self, n_classes: int):
        self.n_classes = n_classes

    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,  # noqa: ARG002
    ) -> Logits:
        # Return random logits with the same batch size as masked_features
        batch_size = masked_features.shape[0]
        logits = torch.ones(batch_size, self.n_classes)

        return logits

    def save(self, path: Path) -> None:
        """Save the classifier to a file. Only n_classes needs to be stored."""
        torch.save(self.n_classes, path)

    @staticmethod
    def load(path: str, device: torch.device) -> "UniformDummyAFAClassifier":
        """Load the classifier from a file, placing it on the given device."""
        # Load the number of classes
        n_classes = torch.load(path, map_location=device)

        # Return a new DummyClassifier instance
        return UniformDummyAFAClassifier(n_classes)


class Predictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
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
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        # Classifier expects flat features, so flatten them
        assert feature_shape is not None
        masked_features_flat = masked_features.flatten(
            start_dim=-len(feature_shape)
        )
        feature_mask_flat = feature_mask.flatten(start_dim=-len(feature_shape))

        with torch.no_grad():
            logits = self.module(masked_features_flat, feature_mask_flat)
        return logits.softmax(dim=-1).to(original_device)

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.module.state_dict(),
                "n_features": self.module.n_features,
                "n_classes": self.module.n_classes,
                "num_cells": self.module.num_cells,
                "dropout": self.module.dropout,
            },
            path / "model.pt",
        )

    @override
    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(
            path / "model.pt", map_location=device, weights_only=False
        )
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
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
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
        path.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.module.state_dict(),
            "num_classes": int(self.module.fc.out_features),
            "pretrained_model_name": self.pretrained_model_name,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
        }
        torch.save(checkpoint, path / "model.pt")

    @override
    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(
            path / "model.pt", map_location=device, weights_only=False
        )
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


def _to_numpy(tensor: torch.Tensor, n_feature_dims: int) -> np.ndarray:
    """Flatten trailing feature dims and convert a tensor to a numpy array."""
    return tensor.detach().to("cpu").flatten(start_dim=-n_feature_dims).numpy()


@final
class WrappedMALearnClassifier(AFAClassifier):
    """A sklearn-based MA classifier wrapped for the AFA classifier protocol."""

    def __init__(
        self,
        model: Any,
        model_name: str,
        n_classes: int,
        device: torch.device,
    ):
        self.model = model
        self.model_name = model_name
        self.n_classes = int(n_classes)
        self._device = device

    def _predict_proba_with_mask(
        self,
        X: np.ndarray,
        missing_mask: np.ndarray,
    ) -> np.ndarray:
        if self.model_name == "malasso":
            transformed = self.model._transform_input(  # pyright: ignore[reportAttributeAccessIssue]
                X,
                missing_mask,
            )
            return self.model.predict_proba(transformed)

        if self.model_name == "madt":
            return self.model.predict(
                X,
                M=missing_mask,
                return_proba=True,
            )

        # MARF and MAGBT do not take M at inference time.
        return self.model.predict_proba(X)

    def _expand_missing_classes(self, probs: np.ndarray) -> np.ndarray:
        if probs.shape[1] == self.n_classes:
            return probs

        classes = getattr(self.model, "classes_", None)
        if classes is None:
            msg = (
                "Model probabilities have fewer classes than expected and "
                "no class mapping was found."
            )
            raise ValueError(msg)

        expanded = np.zeros((probs.shape[0], self.n_classes), dtype=np.float64)
        expanded[:, np.asarray(classes, dtype=int)] = probs
        return expanded

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        _label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        original_device = masked_features.device
        if feature_shape is None:
            msg = "feature_shape is required for WrappedMALearnClassifier."
            raise ValueError(msg)

        n_feature_dims = len(feature_shape)
        masked_features_flat = _to_numpy(masked_features, n_feature_dims)
        feature_mask_flat = _to_numpy(feature_mask, n_feature_dims).astype(
            bool
        )

        X = masked_features_flat.copy()
        X[~feature_mask_flat] = 0.0
        missing_mask = (~feature_mask_flat).astype(np.int8)

        probs = self._predict_proba_with_mask(X=X, missing_mask=missing_mask)
        probs = self._expand_missing_classes(probs)

        probs_t = torch.from_numpy(probs.astype(np.float32, copy=False))
        return probs_t.to(original_device)

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model,
                "model_name": self.model_name,
                "n_classes": self.n_classes,
            },
            path / "model.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(
            path / "model.pt",
            map_location="cpu",
            weights_only=False,
        )
        return cls(
            model=checkpoint["model"],
            model_name=str(checkpoint["model_name"]),
            n_classes=int(checkpoint["n_classes"]),
            device=device,
        )

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        return self
