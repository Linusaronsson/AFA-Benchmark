import csv
import logging
from collections.abc import Sequence
from math import prod
from pathlib import Path
from typing import Protocol, Self

import torch
from jaxtyping import Bool, Float, Integer

logger = logging.getLogger(__name__)

type Features = Float[torch.Tensor, "*batch *feature_shape"]
# MaskedFeatures are similar to Features, but are 0 where FeatureMask is False
type MaskedFeatures = Float[torch.Tensor, "*batch *feature_shape"]
type FeatureMask = Bool[torch.Tensor, "*batch *feature_shape"]
type SelectionMask = Bool[torch.Tensor, "*batch *selection_shape"]
# We allow arbitrary labels
type Label = Float[torch.Tensor, "*batch *label_shape"]
type Logits = Float[torch.Tensor, "*batch *n_classes"]


# Outputs of AFA methods, representing which feature group to acquire next, or to stop acquiring features (0)
type AFAAction = Integer[torch.Tensor, "*batch 1"]

# Similar to an AFAAction, but does not include the stop action!
# Unmaskers receive this.
type AFASelection = Integer[torch.Tensor, "*batch 1"]

_DATASET_KEY_ALIASES = {
    "ACTG175Dataset": "actg",
    "AFAContextDataset": "afa_context",
    "BankMarketingDataset": "bank_marketing",
    "CKDDataset": "ckd",
    "CubeDataset": "cube",
    "DiabetesDataset": "diabetes",
    "FICODataset": "fico",
    "FashionMNISTDataset": "fashion_mnist",
    "ImagenetteDataset": "imagenette",
    "MiniBooNEDataset": "miniboone",
    "MNISTDataset": "mnist",
    "PharyngitisDataset": "pharyngitis",
    "PhysionetDataset": "physionet",
    "SyntheticMNISTDataset": "synthetic_mnist",
}


def _infer_dataset_key(dataset: object) -> str:
    class_name = dataset.__class__.__name__
    if class_name in _DATASET_KEY_ALIASES:
        return _DATASET_KEY_ALIASES[class_name]
    class_name = class_name.removesuffix("Dataset")
    key_parts = []
    for idx, char in enumerate(class_name):
        if char.isupper() and idx > 0:
            key_parts.append("_")
        key_parts.append(char.lower())
    return "".join(key_parts)


class AFADataset(Protocol):
    """
    Datasets that can be used for evaluating AFA methods.

    The constructor should generate data immediately. For deterministic loading,
    use the load() class method which bypasses __init__ using __new__.

    If the dataset is synthetic, accepts_seed() should return True. In that case, the constructor should also accept a `seed` argument.
    """

    feature_costs: torch.Tensor | None

    @property
    def feature_shape(self) -> torch.Size:
        """Return the shape of features (excluding batch dimension)."""
        ...

    @property
    def label_shape(self) -> torch.Size:
        """Return the shape of labels (excluding batch dimension)."""
        ...

    @classmethod
    def accepts_seed(cls) -> bool:
        """Return whether the dataset constructor accepts a seed parameter."""
        ...

    def create_subset(self, indices: Sequence[int]) -> Self:
        """
        Return a new dataset instance containing only the specified indices.

        Implementers must provide this method. For in-memory datasets with
        `features` and `labels` attributes, you may use the `default_create_subset` function.
        """
        ...

    def __getitem__(self, idx: int) -> tuple[Features, Label]:
        """Return a single sample from the dataset as (features, label)."""
        ...

    def __len__(self) -> int: ...

    def get_all_data(
        self,
    ) -> tuple[Features, Label]:
        """Return all of the data in the dataset as (features, labels). Useful for testing purposes to compare dataset contents."""
        ...

    def save(self, path: Path) -> None:
        """Save the dataset to a file or folder. The file/folder should be in a format that can be loaded by the dataset. This enables deterministic loading of datasets."""
        ...

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load the dataset from a file/folder. The file/folder should contain the dataset in a format that can be loaded by the dataset. This enables deterministic loading of datasets."""
        ...

    def get_feature_acquisition_costs(self) -> torch.Tensor:
        """Return the acquisition costs for each feature as a tensor of the shape as the features."""
        feature_costs = getattr(self, "feature_costs", None)
        if feature_costs is None:
            feature_costs = self.load_feature_costs()
            self.feature_costs = feature_costs
        return feature_costs

    def load_feature_costs(
        self,
        dataset_key: str | None = None,
        feature_shape: torch.Size | None = None,
        costs_path: Path | None = None,
    ) -> torch.Tensor:
        """
        Load per-feature acquisition costs from a CSV file if it exists.

        Falls back to uniform unit costs when no file is present.
        """
        resolved_shape = (
            feature_shape if feature_shape is not None else self.feature_shape
        )
        resolved_key = dataset_key or _infer_dataset_key(self)
        resolved_path = (
            costs_path
            or Path("extra/data/misc/feature_costs") / f"{resolved_key}.csv"
        )
        if not resolved_path.exists():
            logger.info(
                "No feature cost CSV for %s at %s. Using unit costs.",
                resolved_key,
                resolved_path,
            )
            return torch.ones(resolved_shape, dtype=torch.float32)
        values: list[float] = []
        with resolved_path.open(newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                for cell in row:
                    cell_value = cell.strip()
                    if cell_value:
                        values.append(float(cell_value))
        expected = prod(resolved_shape)
        assert len(values) == expected, (
            f"Feature cost file {resolved_path} has {len(values)} values, "
            f"expected {expected} for feature shape {resolved_shape}."
        )
        costs = torch.tensor(values, dtype=torch.float32).reshape(
            resolved_shape
        )
        logger.info(
            "Loaded feature costs for %s from %s (n=%d, min=%.4f, max=%.4f, mean=%.4f).",
            resolved_key,
            resolved_path,
            costs.numel(),
            costs.min().item(),
            costs.max().item(),
            costs.mean().item(),
        )
        return costs


class AFAMethod(Protocol):
    """An AFA method is an object that can decide which features to collect next (or stop collecting features) and also do predictions with the features it has seen so far."""

    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        """
        Return an AFA action based on observed features (1-indexed selection or 0 to stop).

        Note that the features in general can have *any* feature shape. If your model expects flat features, use `feature_shape` to flatten them first.

        Args:
            masked_features: The features with unobserved features masked out (set to zero).
            feature_mask: A boolean mask indicating which features have been observed.
            selection_mask: A boolean mask indicating which selections have already been performed.
            label: The true label, if available (may be None during inference). We include this possibility to support "cheating" methods for benchmarking purposes.
            feature_shape: The shape of the features excluding the batch dimension, if needed.
        """
        ...

    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        Return the predicted label for the features that have been observed so far.

        Args:
            masked_features: The features with unobserved features masked out (set to zero).
            feature_mask: A boolean mask indicating which features have been observed.
            label: The true label, if available (may be None during inference). We include this possibility to support "cheating" methods for benchmarking purposes.
            feature_shape: The shape of the features excluding the batch dimension, if needed. Since both masked_features and label can have multiple batch dimensions, the feature shape cannot be inferred automatically.
        """
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
        return False

    @property
    def cost_param(self) -> float | None:
        """Return the cost parameter, if any. Only applies to methods that make trade-offs between feature cost and accuracy."""
        return None

    def set_cost_param(self, cost_param: float) -> None:
        """Set the cost parameter, if any. Mostly applies to methods that do not need a cost parameter during training but can adjust the trade-off during evaluation."""
        pass  # noqa: PIE790

    def set_seed(self, seed: int | None) -> None:
        """Set the seed, if the method is stochastic."""
        pass  # noqa: PIE790


class AFAClassifier(Protocol):
    """
    An AFA classifier is an object that can perform classification on masked features.

    Classifiers saved as artifacts should follow this protocol to ensure compatibility with the evaluation scripts.
    """

    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """
        Return the predicted label for the features that have been observed so far.

        Note that the features in general can have *any* feature shape. If your model expects flat features, use `feature_shape` to flatten them first.

        Args:
            masked_features: The features with unobserved features masked out (set to zero).
            feature_mask: A boolean mask indicating which features have been observed.
            label: The true label, if available (may be None during inference). We include this possibility to support "cheating" methods for benchmarking purposes.
            feature_shape: The shape of the features excluding the batch dimension, if needed. Since both masked_features and label can have multiple batch dimensions, the feature shape cannot be inferred automatically.
        """
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


# Feature action interface assumed during evaluation
class AFAActionFn(Protocol):
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        """
        Make a new AFA action (1-indexed selection or 0 to stop).

        Note that the features in general can have *any* feature shape. If your model expects flat features, use `feature_shape` to flatten them first.

        Args:
            masked_features: The features with unobserved features masked out (set to zero).
            feature_mask: A boolean mask indicating which features have been observed.
            selection_mask: A boolean mask indicating which selections have already been performed. Note that a selection is generally not the same as a feature.
            label: The true label, if available (may be None during inference). We include this possibility to support "cheating" methods for benchmarking purposes.
            feature_shape: The shape of the features excluding the batch dimension, if needed. Since both masked_features and label can have multiple batch dimensions, the feature shape cannot be inferred automatically.
        """
        ...


# Classifier prediction interface assumed during evaluation
class AFAPredictFn(Protocol):
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        """
        Make a prediction.

        Note that the features in general can have *any* feature shape. If your model expects flat features, use `feature_shape` to flatten them first.

        Args:
            masked_features: The features with unobserved features masked out (set to zero).
            feature_mask: A boolean mask indicating which features have been observed.
            label: The true label, if available (may be None during inference). We include this possibility to support "cheating" methods for benchmarking purposes.
            feature_shape: The shape of the features excluding the batch dimension, if needed. Since both masked_features and label can have multiple batch dimensions, the feature shape cannot be inferred automatically.
        """
        ...


class AFAUnmasker(Protocol):
    def get_n_selections(self, feature_shape: torch.Size) -> int:
        """Return how many different selections are possible with this unmasker."""
        ...

    def get_selection_costs(self, feature_costs: torch.Tensor) -> torch.Tensor:
        """
        Return selection costs given feature costs.

        For example, a patch-based image unmasker might return the summed feature costs within each patch.

        Args:
            feature_costs (torch.Tensor): How much each feature in a dataset costs. This tensor is expected to have the same shape as the features themselves.
        """
        ...

    def unmask(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        afa_selection: AFASelection,
        selection_mask: SelectionMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        """
        Unmask features, given a selection.

        Args:
            masked_features: The features with unobserved features masked out (set to zero).
            feature_mask: A boolean mask indicating which features have been observed.
            features: The original features before masking.
            afa_selection: The AFA selection indicating which feature group to unmask next (does not include stop action).
            selection_mask: A boolean mask indicating which selections have already been performed. Note that a selection is generally not the same as a feature.
            label: The true label, if available (may be None during inference). We include this possibility to support "cheating" methods for benchmarking purposes.
            feature_shape: The shape of the features excluding the batch dimension, if needed. Since both masked_features and label can have multiple batch dimensions, the feature shape cannot be inferred automatically.
        """
        ...

    def set_seed(self, seed: int | None) -> None: ...


class AFAUnmaskFn(Protocol):
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        features: Features,
        afa_selection: AFASelection,
        selection_mask: SelectionMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        """
        Unmask features, given a selection.

        Args:
            masked_features: The features with unobserved features masked out (set to zero).
            feature_mask: A boolean mask indicating which features have been observed.
            features: The original features before masking.
            afa_selection: The AFA selection indicating which feature group to unmask next (does not include stop action).
            selection_mask: A boolean mask indicating which selections have already been performed. Note that a selection is generally not the same as a feature.
            label: The true label, if available (may be None during inference). We include this possibility to support "cheating" methods for benchmarking purposes.
            feature_shape: The shape of the features excluding the batch dimension, if needed. Since both masked_features and label can have multiple batch dimensions, the feature shape cannot be inferred automatically.
        """
        ...


class AFAInitializer(Protocol):
    def set_seed(self, seed: int | None) -> None: ...

    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        """
        Create initial feature mask.

        Args:
            features: The original features before masking.
            label: The true label, if available (may be None during inference). We include this possibility to support "cheating" methods for benchmarking purposes.
            feature_shape: The shape of the features excluding the batch dimension, if needed. Since both masked_features and label can have multiple batch dimensions, the feature shape cannot be inferred automatically.
        """
        ...

    def get_training_forbidden_mask(
        self,
        observed_mask: FeatureMask,
    ) -> FeatureMask:
        """
        Return mask of features permanently unavailable during training.

        When training AFA methods on incomplete data, this mask indicates
        which features should never be visible or selectable. The training
        loop should start from a cold state and block these features.

        Args:
            observed_mask: Boolean mask from ``initialize()`` where
                True=observed. Shape: ``(*batch, *feature_shape)``.

        Returns:
            Boolean mask where True=forbidden (permanently unavailable).
                Same shape as ``observed_mask``.
                Default: all False (nothing forbidden).
        """
        return torch.zeros_like(observed_mask, dtype=torch.bool)


class AFAInitializeFn(Protocol):
    def __call__(
        self,
        features: Features,
        label: Label,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        """
        Create initial feature mask.

        Args:
            features: The original features before masking.
            label: The true label, if available (may be None during inference).
            feature_shape: The shape of the features excluding the batch dimension, if needed. Since both masked_features and label can have multiple batch dimensions, the feature shape cannot be inferred automatically.
        """
        ...
