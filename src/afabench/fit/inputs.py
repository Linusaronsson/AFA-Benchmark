"""Lazy access to the inputs that a contract points to."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch

from afabench.components.initializers.utils import (
    get_afa_initializer_from_config,
)
from afabench.components.unmaskers.utils import get_afa_unmasker_from_config
from afabench.core.bundle_system.bundle import (
    Manifest,
    load_bundle_from_manifest,
)
from afabench.core.provenance import (
    DatasetIdentity,
    InputRole,
    ProvenanceInput,
    shared_dataset_identity,
)
from afabench.core.types import AFADataset, AFAInitializer, AFAUnmasker
from afabench.fit.contract import BaseContract, TrainingContract


class UnavailableFitInputError(LookupError):
    """Raised when the contract does not provide the requested input."""


class FitInputs:
    """
    Inputs of a contract, each loaded on first request.

    Initializer and Unmasker are seeded with the contract's seed. Classifier
    and pretrained model are loaded onto the contract's device and checked
    against the class the caller expects. Each input bundle's
    manifest is read at most once, whether for loading the bundle or for
    the dataset identity and provenance inputs of the fit-stage result.
    """

    def __init__(self, contract: BaseContract) -> None:
        self._contract: BaseContract = contract
        self._loaded: dict[str, object] = {}
        self._manifests: dict[str, Manifest] = {}
        self._bundles: dict[tuple[str, type], object] = {}

    def train_dataset(self) -> AFADataset:
        return self._load_dataset(self._contract.train_dataset_bundle_path)

    def val_dataset(self) -> AFADataset:
        return self._load_dataset(self._contract.val_dataset_bundle_path)

    def initializer(self) -> AFAInitializer:
        initializer = self._load_once("initializer", self._seeded_initializer)
        return cast("AFAInitializer", initializer)

    def unmasker(self) -> AFAUnmasker:
        unmasker = self._load_once("unmasker", self._seeded_unmasker)
        return cast("AFAUnmasker", unmasker)

    def classifier[T](self, expected_type: type[T]) -> T:
        return self._load_on_device(
            self._contract.classifier_bundle_path, expected_type
        )

    def pretrained_model[T](self, expected_type: type[T]) -> T:
        if not isinstance(self._contract, TrainingContract):
            msg = "The pretraining stage provides no pretrained model."
            raise UnavailableFitInputError(msg)
        path = self._contract.pretrained_model_bundle_path
        if path is None:
            msg = "No pretrained model: pretrained_model_bundle_path is null."
            raise UnavailableFitInputError(msg)
        return self._load_on_device(path, expected_type)

    def dataset_identity(self) -> DatasetIdentity:
        """
        Return the dataset identity the train and val dataset records share.

        Raises `DatasetIdentityMismatchError` if they disagree.
        """
        return shared_dataset_identity(
            self._manifest(
                self._contract.train_dataset_bundle_path
            ).provenance,
            self._manifest(self._contract.val_dataset_bundle_path).provenance,
        )

    def provenance_inputs(self) -> list[ProvenanceInput]:
        """Describe the contract's input bundles as provenance inputs."""
        paths: list[tuple[InputRole, str]] = [
            ("train_dataset", self._contract.train_dataset_bundle_path),
            ("val_dataset", self._contract.val_dataset_bundle_path),
            ("classifier", self._contract.classifier_bundle_path),
        ]
        if (
            isinstance(self._contract, TrainingContract)
            and self._contract.pretrained_model_bundle_path is not None
        ):
            paths.append(
                (
                    "pretrained_model",
                    self._contract.pretrained_model_bundle_path,
                )
            )
        return [
            self._manifest(path).as_input(role, path) for role, path in paths
        ]

    def _seeded_initializer(self) -> AFAInitializer:
        initializer = get_afa_initializer_from_config(
            self._contract.initializer
        )
        initializer.set_seed(self._contract.seed)
        return initializer

    def _seeded_unmasker(self) -> AFAUnmasker:
        unmasker = get_afa_unmasker_from_config(self._contract.unmasker)
        unmasker.set_seed(self._contract.seed)
        return unmasker

    def _manifest(self, bundle_path: str) -> Manifest:
        if bundle_path not in self._manifests:
            self._manifests[bundle_path] = Manifest.read(Path(bundle_path))
        return self._manifests[bundle_path]

    def _load_dataset(self, bundle_path: str) -> AFADataset:
        return self._load_bundle(bundle_path, AFADataset)

    def _load_on_device[T](
        self, bundle_path: str, expected_type: type[T]
    ) -> T:
        return self._load_bundle(
            bundle_path,
            expected_type,
            device=torch.device(self._contract.device),
        )

    def _load_bundle[T](
        self,
        bundle_path: str,
        expected_type: type[T],
        **kwargs: object,
    ) -> T:
        # Keyed by the expected class too, so that every request is checked
        # against its own class by `load_bundle_from_manifest`
        key = (bundle_path, expected_type)
        if key not in self._bundles:
            self._bundles[key] = load_bundle_from_manifest(
                bundle_path,
                self._manifest(bundle_path),
                expect=expected_type,
                **kwargs,
            ).obj
        return cast("T", self._bundles[key])

    def _load_once(self, key: str, load: Callable[[], object]) -> object:
        if key not in self._loaded:
            self._loaded[key] = load()
        return self._loaded[key]


def load_inputs(contract: BaseContract) -> FitInputs:
    return FitInputs(contract)
