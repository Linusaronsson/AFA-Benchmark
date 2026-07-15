"""General wrapper for PyTorch models to work with the bundle system."""

from pathlib import Path
from typing import Any, Self, override

import torch
from torch import nn

from afabench.core.bundle_system.bundle import Loadeable, Saveable


class TorchModelBundle(Saveable, Loadeable):
    """
    A general wrapper for PyTorch models that implements Saveable/Loadeable protocols.

    This allows any nn.Module to be saved and loaded using the bundle system.
    """

    _class_version: str = "1.0.0"

    def __init__(self, model: nn.Module):
        self.model: nn.Module = model

    @override
    def save(self, path: Path) -> None:
        """Save the PyTorch model to the specified path."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, path / "model.pt")

    @classmethod
    @override
    def load(cls, path: Path, **kwargs: Any) -> Self:
        """
        Load a PyTorch model from the specified path.

        Args:
            path: Path to the saved model
            **kwargs: Arbitrary keyword arguments. Supports 'device' for target device.
                     Uses Any because bundle system must support any object type.
        """
        device = kwargs.get("device", torch.device("cpu"))
        model = torch.load(
            path / "model.pt", map_location=device, weights_only=False
        )
        return cls(model)

    def to(self, device: torch.device) -> Self:
        """Move the wrapped model to the specified device."""
        self.model = self.model.to(device)
        return self

    @property
    def device(self) -> torch.device:
        """Get the device of the wrapped model."""
        return next(self.model.parameters()).device
