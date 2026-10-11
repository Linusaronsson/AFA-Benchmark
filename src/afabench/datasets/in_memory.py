"""
The in-memory base behind the AFADataset classes.

A dataset that holds all its instances as tensors in memory subclasses
`InMemoryAFADataset` and supplies its shapes, the constructor arguments
that are its config, and how to build its tensors. The base owns the rest
of the `AFADataset` protocol: subsets, indexing, and the `dataset.pt` file
each bundle stores, with the generation indices (ADR 0004) and feature
costs (ADR 0011) every dataset must persist.
"""

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Self, override

import torch
from torch import Tensor
from torch.utils.data import Dataset

from afabench.core.types import (
    AFADataset,
    Features,
    GenerationIndices,
    Label,
)
from afabench.datasets.feature_costs import load_saved_feature_costs
from afabench.datasets.utils import load_generation_indices

DATASET_FILENAME = "dataset.pt"


class MissingConfigKeyError(ValueError):
    """A saved dataset's config lacks a key its class declares."""


class InMemoryAFADataset(Dataset[tuple[Tensor, Tensor]], AFADataset, ABC):
    """
    Base of the AFADataset classes whose instances are tensors in memory.

    A subclass declares `feature_shape`, `label_shape`, `accepts_seed` and
    `config_keys`, sets its config attributes in its constructor before
    calling `super().__init__()`, and returns its tensors from `_build`.
    """

    # Constructor arguments that are the dataset's config: `save` persists
    # them under "config" in dataset.pt and `load` restores them as
    # attributes of the same name
    config_keys: ClassVar[tuple[str, ...]]

    features: Features
    labels: Label
    generation_indices: GenerationIndices
    feature_costs: Tensor | None = None

    @property
    @abstractmethod
    @override
    def feature_shape(self) -> torch.Size: ...

    @property
    @abstractmethod
    @override
    def label_shape(self) -> torch.Size: ...

    @classmethod
    @abstractmethod
    @override
    def accepts_seed(cls) -> bool: ...

    def __init__(self) -> None:
        super().__init__()
        features, labels = self._build()
        self.features = features
        self.labels = labels
        self._check_shapes()
        self.generation_indices = torch.arange(
            len(self.features), dtype=torch.long
        )

    def _check_shapes(self) -> None:
        for name, tensor, shape in (
            ("features", self.features, self.feature_shape),
            ("labels", self.labels, self.label_shape),
        ):
            if tensor.shape[1:] != shape:
                msg = (
                    f"{type(self).__name__} built {name} of shape "
                    f"{tuple(tensor.shape[1:])} per instance, expected "
                    f"{tuple(shape)}"
                )
                raise ValueError(msg)
        if len(self.labels) != len(self.features):
            msg = (
                f"{type(self).__name__} built {len(self.labels)} labels for "
                f"{len(self.features)} instances"
            )
            raise ValueError(msg)

    @abstractmethod
    def _build(self) -> tuple[Features, Label]:
        """
        Return the features and labels of every instance, in generation order.

        For a real-world dataset that is the source's order (ADR 0004).
        """

    @property
    def config(self) -> dict[str, object]:
        """The constructor arguments that generated the data."""
        return {key: getattr(self, key) for key in self.config_keys}

    @override
    def create_subset(self, indices: Sequence[int]) -> Self:
        positions = list(indices)
        subset = copy.copy(self)
        subset.features = self.features[positions]
        subset.labels = self.labels[positions]
        subset.generation_indices = self.generation_indices[positions]
        return subset

    @override
    def get_generation_indices(self) -> GenerationIndices:
        return self.generation_indices

    @override
    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        return self.features[idx], self.labels[idx]

    @override
    def __len__(self) -> int:
        return len(self.features)

    @override
    def get_all_data(self) -> tuple[Features, Label]:
        return self.features, self.labels

    def _saved_data(self) -> dict[str, object]:
        """Return what `save` writes to dataset.pt; a subclass may add entries beside these."""
        return {
            "features": self.features,
            "labels": self.labels,
            "generation_indices": self.generation_indices,
            "feature_costs": self.get_feature_acquisition_costs(),
            "config": self.config,
        }

    @override
    def save(self, path: Path) -> None:
        torch.save(self._saved_data(), path / DATASET_FILENAME)

    @classmethod
    @override
    def load(cls, path: Path) -> Self:
        file = path / DATASET_FILENAME
        data: Mapping[str, object] = torch.load(file)
        # Restore without calling __init__, which would build the data anew
        obj = cls.__new__(cls)
        obj._restore(data, file)  # noqa: SLF001
        return obj

    def _restore(self, data: Mapping[str, object], file: Path) -> None:
        """Set this fresh object's state from the `data` saved in `file`; a subclass restoring more calls this first."""
        config = data.get("config")
        if not isinstance(config, Mapping):
            msg = f"Dataset saved at {file} has no config, got {config!r}"
            raise MissingConfigKeyError(msg)
        for key in self.config_keys:
            if key not in config:
                msg = (
                    f"Config saved at {file} has no {key!r}, which "
                    f"{type(self).__name__} declares; got {dict(config)!r}"
                )
                raise MissingConfigKeyError(msg)
            setattr(self, key, config[key])
        self.features = _saved_tensor(data, "features", file)
        self.labels = _saved_tensor(data, "labels", file)
        self.generation_indices = load_generation_indices(
            data, file, len(self.features)
        )
        self.feature_costs = load_saved_feature_costs(
            data, file, self.feature_shape
        )


def _saved_tensor(data: Mapping[str, object], key: str, file: Path) -> Tensor:
    tensor = data.get(key)
    if not isinstance(tensor, Tensor):
        msg = f"Dataset saved at {file} has no {key} tensor, got {tensor!r}"
        raise TypeError(msg)
    return tensor
