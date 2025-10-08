from pathlib import Path
from typing import ClassVar, Protocol, Self

import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

# AFA datasets return features and labels
type Features = Float[Tensor, "*batch n_features"]
# We use float here since in general we can have probabilities, not only one-hot
type Label = Float[Tensor, "*batch n_classes"]
# We need to be able to distinguish between samples, e.g., for tracking performance per sample
type SampleIndex = Integer[Tensor, "*batch 1"]

type Logits = Float[Tensor, "*batch model_output_size"]


class AFADataset(Protocol):
    """
    Datasets that can be used for evaluating AFA methods.

    Notably, the __init__ constructor should *not* generate data. Instead, generate_data() should be called. This makes it possible to call load if deterministic data is desired.
    """

    # Used by AFADatasetFn
    features: Features  # batched
    labels: Label  # batched
    indices: SampleIndex

    # Used by evaluation scripts to avoid loading the dataset
    n_classes: ClassVar[int]
    n_features: ClassVar[int]

    def generate_data(self) -> None:
        """Generate data."""
        ...

    def __getitem__(self, idx: int) -> tuple[Features, Label, SampleIndex]:
        """Return a single sample from the dataset. The index of the sample in the dataset should also be returned."""
        ...

    def __len__(self) -> int: ...

    def get_all_data(
        self,
    ) -> tuple[Features, Label, SampleIndex]:
        """Return all of the data in the dataset. Useful for batched computations."""
        ...

    def save(self, path: Path) -> None:
        """Save the dataset to a file. The file should be in a format that can be loaded by the dataset. This enables deterministic loading of datasets."""
        ...

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load the dataset from a file. The file should contain the dataset in a format that can be loaded by the dataset. This enables deterministic loading of datasets."""
        ...


type MaskedFeatures = Integer[Tensor, "*batch n_features"]
type FeatureMask = Bool[Tensor, "*batch n_features"]

# Outputs of AFA methods, representing which feature to collect next, or to stop acquiring features (0)
type AFASelection = Integer[Tensor, "*batch 1"]


class AFAMethod(Protocol):
    """An AFA method is an object that can decide which features to collect next (or stop collecting features) and also do predictions with the features it has seen so far."""

    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None,
        label: Label | None,
    ) -> AFASelection:
        """Return the 0-based index of the feature to be collected."""
        ...

    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None,
        label: Label | None,
    ) -> Label:
        """Return the predicted label for the features that have been observed so far."""
        ...

    def save(self, path: Path) -> None:
        """Save the method to disk. The folder should be in a format that can be loaded by the method."""
        ...

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the method from a folder, placing it on the given device."""
        ...

    def to(self, device: torch.device) -> Self:
        """Move the object to the specified device. This should determine on which device calculations will be performed."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the current device the method is on."""
        ...

    @property
    def has_builtin_classifier(self) -> bool:
        """Return the current device the method is on."""
        ...

    @property
    def cost_param(self) -> float | None:
        """Return the cost parameter, if any. Only applies to methods that make trade-offs between feature cost and accuracy."""
        ...

    def set_cost_param(self, cost_param: float) -> None:
        """Set the cost parameter, if any. Mostly applies to methods that do not need a cost parameter during training but can adjust the trade-off during evaluation."""
        ...


class AFAClassifier(Protocol):
    """
    An AFA classifier is an object that can perform classification on masked features.

    Classifiers saved as artifacts should follow this protocol to ensure compatibility with the evaluation scripts.
    """

    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None,
        label: Label | None,
    ) -> Label:
        """Return the predicted label for the features that have been observed so far."""
        ...

    def save(self, path: Path) -> None:
        """Save the classifier to a file or folder. The file/folder should be in a format that can be loaded by the method."""
        ...

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the classifier from a file or folder, placing it on the given device."""
        ...

    def to(self, device: torch.device) -> Self:
        """Move the object to the specified device. This should determine on which device calculations will be performed."""
        ...

    @property
    def device(self) -> torch.device:
        """Return the current device the method is on."""
        ...


# Feature selection interface assumed during evaluation
class AFASelectFn(Protocol):
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None,
        label: Label | None,
    ) -> AFASelection: ...


# Classifier prediction interface assumed during evaluation
class AFAPredictFn(Protocol):
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features | None,
        label: Label | None,
    ) -> Label: ...
