"""
The implementation base behind the AFAMethod protocol.

`AFAMethod` in `afabench.core.types` is a pure protocol: what the evaluator
and the bundle system call, with no default bodies. A method class
subclasses `AFAMethodBase`, which owns what every method repeated: the
device it computes on and `to`, the round trip of `act` and `predict`
inputs onto that device and outputs back to the caller's
(`on_method_device`), and the defaults of a method with no soft-budget
parameter and no generator of its own.

Each method keeps its own `save` and `load`, so bundles written before
this base load unchanged; `save_classifier` and `load_classifier` share
the one layout several methods write for the classifier they ship.
"""

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Concatenate, Self, cast, override

import torch
from torch import Tensor, nn

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

CLASSIFIER_FILENAME = "classifier.pt"
CLASSIFIER_CLASS_NAME_FILENAME = "classifier_class_name.txt"


class AFAMethodBase(AFAMethod, ABC):
    """
    Base of the AFAMethod classes.

    A subclass sets its device with `super().__init__(device)`, or as a
    dataclass declares the `_device` field; declares
    `has_builtin_classifier`; implements `act`, `predict`, `save` and
    `load`; and extends `to` for state the base cannot see, such as modules
    held in a container.
    """

    _device: torch.device

    def __init__(self, device: torch.device) -> None:
        self._device = device

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @override
    def to(self, device: torch.device) -> Self:
        """Move every tensor, module and classifier attribute to `device`, and compute there from now on."""
        self._device = device
        for name, value in vars(self).items():
            if isinstance(value, Tensor | nn.Module | AFAClassifier):
                setattr(self, name, value.to(device))
        return self

    @property
    @abstractmethod
    @override
    def has_builtin_classifier(self) -> bool: ...

    @abstractmethod
    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction: ...

    @abstractmethod
    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor: ...

    @abstractmethod
    @override
    def save(self, path: Path) -> None: ...

    @classmethod
    @abstractmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self: ...

    @property
    @override
    def cost_param(self) -> float | None:
        """The soft-budget parameter; None for a method without one."""
        return None

    @override
    def set_cost_param(self, cost_param: float) -> None:
        """Set the soft-budget parameter; a method without one ignores it."""

    @override
    def set_seed(self, seed: int | None) -> None:
        """
        Seed the method's own generator, if it has one.

        The evaluator seeds torch's global generators before calling this,
        which covers a method that draws from them.
        """


def on_method_device[M: AFAMethodBase, **P](
    fn: Callable[Concatenate[M, P], Tensor],
) -> Callable[Concatenate[M, P], Tensor]:
    """
    Run `act` or `predict` on the method's device and answer on the caller's.

    Every tensor argument is moved to `self.device` before the body runs,
    and the result is moved to the device of `masked_features`, passed by
    position or by keyword.
    """
    # Rebuilt argument tuples cannot be typed as P.args, so the wrapper
    # calls through an untyped alias; callers see fn's own signature
    untyped_fn = cast("Callable[..., Tensor]", fn)

    @functools.wraps(fn)
    def wrapper(self: M, *args: P.args, **kwargs: P.kwargs) -> Tensor:
        caller_device = _masked_features(args, kwargs).device
        moved_args = tuple(_on_device(arg, self.device) for arg in args)
        moved_kwargs = {
            name: _on_device(value, self.device)
            for name, value in kwargs.items()
        }
        return untyped_fn(self, *moved_args, **moved_kwargs).to(caller_device)

    return wrapper


def _masked_features(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> Tensor:
    value = args[0] if args else kwargs.get("masked_features")
    if not isinstance(value, Tensor):
        msg = (
            "on_method_device needs masked_features as a tensor to answer "
            f"on its device, got {type(value).__name__}"
        )
        raise TypeError(msg)
    return value


def _on_device(value: object, device: torch.device) -> object:
    if isinstance(value, Tensor):
        return value.to(device)
    return value


def save_classifier(classifier: AFAClassifier, path: Path) -> None:
    """Save a method's classifier under `path` as classifier.pt, with its registered class name beside it."""
    classifier.save(path / CLASSIFIER_FILENAME)
    (path / CLASSIFIER_CLASS_NAME_FILENAME).write_text(
        type(classifier).__name__
    )


def load_classifier(path: Path, device: torch.device) -> AFAClassifier:
    """Load the classifier `save_classifier` wrote under `path` onto `device`."""
    class_name = (path / CLASSIFIER_CLASS_NAME_FILENAME).read_text()
    classifier: AFAClassifier = get_class(class_name).load(
        path / CLASSIFIER_FILENAME, device
    )
    return classifier
