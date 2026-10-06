"""Lazy access to the inputs that the training contract points to."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch

from afabench.components.initializers.utils import (
    get_afa_initializer_from_config,
)
from afabench.components.unmaskers.utils import get_afa_unmasker_from_config
from afabench.core.bundle_system.bundle import load_bundle
from afabench.core.types import AFADataset, AFAInitializer, AFAUnmasker
from afabench.training.contract import PretrainingContract, TrainingContract


class UnavailableTrainingInputError(LookupError):
    """Raised when the contract does not provide the requested input."""


class TrainingInputs:
    """
    Inputs of a training contract, each loaded on first request.

    Classifier and pretrained model are loaded onto the contract's device and
    checked against the class the caller expects.
    """

    def __init__(self, contract: PretrainingContract) -> None:
        self._contract: PretrainingContract = contract
        self._loaded: dict[str, object] = {}

    def train_dataset(self) -> AFADataset:
        return self._load_dataset(self._contract.train_dataset_bundle_path)

    def val_dataset(self) -> AFADataset:
        return self._load_dataset(self._contract.val_dataset_bundle_path)

    def initializer(self) -> AFAInitializer:
        initializer = self._load_once(
            "initializer",
            lambda: get_afa_initializer_from_config(
                self._contract.initializer
            ),
        )
        return cast("AFAInitializer", initializer)

    def unmasker(self) -> AFAUnmasker:
        unmasker = self._load_once(
            "unmasker",
            lambda: get_afa_unmasker_from_config(self._contract.unmasker),
        )
        return cast("AFAUnmasker", unmasker)

    def classifier[T](self, expected_type: type[T]) -> T:
        return self._load_on_device(
            self._contract.classifier_bundle_path, expected_type
        )

    def pretrained_model[T](self, expected_type: type[T]) -> T:
        if not isinstance(self._contract, TrainingContract):
            msg = "The pretraining stage provides no pretrained model."
            raise UnavailableTrainingInputError(msg)
        path = self._contract.pretrained_model_bundle_path
        if path is None:
            msg = "No pretrained model: pretrained_model_bundle_path is null."
            raise UnavailableTrainingInputError(msg)
        return self._load_on_device(path, expected_type)

    def _load_dataset(self, bundle_path: str) -> AFADataset:
        dataset = self._load_once(
            bundle_path, lambda: load_bundle(Path(bundle_path))[0]
        )
        return cast("AFADataset", dataset)

    def _load_on_device[T](
        self, bundle_path: str, expected_type: type[T]
    ) -> T:
        device = torch.device(self._contract.device)
        obj = self._load_once(
            bundle_path,
            lambda: load_bundle(Path(bundle_path), device=device)[0],
        )
        if not isinstance(obj, expected_type):
            msg = (
                f"Expected {expected_type.__name__} in {bundle_path}, "
                f"got {type(obj).__name__}."
            )
            raise TypeError(msg)
        return obj

    def _load_once(self, key: str, load: Callable[[], object]) -> object:
        if key not in self._loaded:
            self._loaded[key] = load()
        return self._loaded[key]


def load_inputs(contract: PretrainingContract) -> TrainingInputs:
    return TrainingInputs(contract)
