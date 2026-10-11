import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from afabench.core.types import AFADataset, GenerationIndices


class MissingGenerationIndicesError(ValueError):
    """A saved dataset has no generation indices (see ADR 0004)."""


def default_create_subset[T: AFADataset](
    dataset: T, indices: Sequence[int]
) -> T:
    """Return a subset of a dataset using default logic for in-memory datasets with .features, .labels and .generation_indices."""
    subset = copy.deepcopy(dataset)
    indices_list = list(indices)
    if (
        hasattr(subset, "features")
        and hasattr(subset, "labels")
        and hasattr(subset, "generation_indices")
    ):
        features = getattr(subset, "features")  # noqa: B009
        labels = getattr(subset, "labels")  # noqa: B009
        generation_indices = getattr(subset, "generation_indices")  # noqa: B009
        setattr(subset, "features", features[indices_list])  # noqa: B010
        setattr(subset, "labels", labels[indices_list])  # noqa: B010
        setattr(  # noqa: B010
            subset, "generation_indices", generation_indices[indices_list]
        )
        if hasattr(subset, "n_samples"):
            setattr(subset, "n_samples", len(indices))  # noqa: B010
    else:
        msg = "default_create_subset requires 'features', 'labels' and 'generation_indices' attributes on the dataset."
        raise AttributeError(msg)
    return subset


def load_generation_indices(
    data: Mapping[str, object], path: Path, n_instances: int | None
) -> "GenerationIndices":
    """
    Return the generation indices saved in a dataset's `data`, loaded from `path`.

    Datasets saved before generation indices existed are not backfilled, so
    their absence is an error rather than a default.
    """
    generation_indices = data.get("generation_indices")
    if generation_indices is None:
        msg = (
            f"Dataset saved at {path} has no generation indices; "
            "regenerate the dataset bundle"
        )
        raise MissingGenerationIndicesError(msg)
    if (
        not isinstance(generation_indices, torch.Tensor)
        or generation_indices.ndim != 1
        or generation_indices.is_floating_point()
    ):
        msg = (
            f"Generation indices saved at {path} must be a 1D integer "
            f"tensor, got {generation_indices!r}"
        )
        raise ValueError(msg)
    if n_instances is not None and len(generation_indices) != n_instances:
        msg = (
            f"Dataset saved at {path} has {len(generation_indices)} "
            f"generation indices for {n_instances} instances"
        )
        raise ValueError(msg)
    return generation_indices.to(dtype=torch.long)


def require_generation_order(dataset: "AFADataset") -> None:
    """
    Require a freshly constructed dataset to number its instances 0..n-1.

    Dataset generation splits by position, so each split's generation
    indices are only positions in the generated dataset if this holds.
    """
    generation_indices = dataset.get_generation_indices()
    if not torch.equal(
        generation_indices.cpu(), torch.arange(len(dataset), dtype=torch.long)
    ):
        msg = (
            f"{type(dataset).__name__} must number its instances 0..n-1 in "
            f"generation order when constructed, got {generation_indices}"
        )
        raise ValueError(msg)
