"""
Feature costs, fixed at dataset generation and stored in dataset bundles.

Dataset generation resolves a dataset's feature costs once: from the
dataset's feature costs file (`data/<dataset key>/feature_costs.csv`) if
there is one, otherwise from the dataset itself, which computes them from
its config or has unit costs. Every split bundle stores the result, and
every later stage reads it from the bundle, never from the file. See
`docs/adr/0011-feature-costs-stored-in-dataset-bundles.md`.
"""

import csv
import logging
from collections.abc import Mapping
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from afabench.core.types import AFADataset

logger = logging.getLogger(__name__)

# Relative to the working directory, which the pipeline runs from the
# repository root (ADR 0010)
DATA_DIR = Path("data")
FEATURE_COSTS_FILENAME = "feature_costs.csv"


class MissingFeatureCostsError(ValueError):
    """A saved dataset has no feature costs (see ADR 0011)."""


class FeatureCostsConflictError(ValueError):
    """A dataset that computes its own feature costs also has a file."""


def feature_costs_path(dataset_key: str, data_dir: Path = DATA_DIR) -> Path:
    """Return where the feature costs file of `dataset_key` would be."""
    return data_dir / dataset_key / FEATURE_COSTS_FILENAME


def check_feature_costs(
    costs: torch.Tensor, feature_shape: torch.Size, source: object
) -> torch.Tensor:
    """Return `costs` as float32 after checking them against `feature_shape`; `source` names where they came from."""
    if costs.shape != feature_shape:
        msg = (
            f"Feature costs from {source} have shape {tuple(costs.shape)}, "
            f"expected the feature shape {tuple(feature_shape)}"
        )
        raise ValueError(msg)
    if not bool(torch.isfinite(costs).all()) or bool((costs < 0).any()):
        msg = (
            f"Feature costs from {source} must be finite and non-negative, "
            f"got {costs}"
        )
        raise ValueError(msg)
    return costs.to(dtype=torch.float32)


def read_feature_costs_file(
    path: Path, feature_shape: torch.Size
) -> torch.Tensor:
    """Read a feature costs CSV: one value per feature, in flattened feature order, split over any rows and columns."""
    values: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            values.extend(float(cell) for cell in row if cell.strip())
    if len(values) != prod(feature_shape):
        msg = (
            f"Feature costs file {path} has {len(values)} values, expected "
            f"{prod(feature_shape)} for feature shape {tuple(feature_shape)}"
        )
        raise ValueError(msg)
    return check_feature_costs(
        torch.tensor(values).reshape(feature_shape), feature_shape, path
    )


def resolve_feature_costs(dataset: "AFADataset", path: Path) -> torch.Tensor:
    """
    Return the feature costs dataset generation stores for `dataset`.

    They are read from the feature costs file at `path` if it exists,
    otherwise taken from the freshly constructed `dataset`. A dataset that
    computes its own feature costs and also has a file raises, since one of
    the two would be silently ignored.
    """
    own_costs = dataset.feature_costs
    if not path.exists():
        costs = dataset.get_feature_acquisition_costs()
        logger.info(
            "No feature costs file at %s; using %s.",
            path,
            "the costs the dataset computes"
            if own_costs is not None
            else "unit costs",
        )
        return check_feature_costs(costs, dataset.feature_shape, path.parent)
    if own_costs is not None:
        msg = (
            f"{type(dataset).__name__} computes its own feature costs from "
            f"its config, so the feature costs file {path} would be ignored; "
            "remove it"
        )
        raise FeatureCostsConflictError(msg)
    costs = read_feature_costs_file(path, dataset.feature_shape)
    logger.info(
        "Read feature costs from %s (min=%.4f, max=%.4f, mean=%.4f).",
        path,
        costs.min().item(),
        costs.max().item(),
        costs.mean().item(),
    )
    return costs


def load_saved_feature_costs(
    data: Mapping[str, object], path: Path, feature_shape: torch.Size
) -> torch.Tensor:
    """
    Return the feature costs saved in a dataset's `data`, loaded from `path`.

    Datasets saved before feature costs were stored are not backfilled from
    a feature costs file, so their absence is an error rather than a default.
    """
    costs = data.get("feature_costs")
    if costs is None:
        msg = (
            f"Dataset saved at {path} has no feature costs; regenerate the "
            "dataset bundle"
        )
        raise MissingFeatureCostsError(msg)
    if not isinstance(costs, torch.Tensor):
        msg = f"Feature costs saved at {path} must be a tensor, got {costs!r}"
        raise TypeError(msg)
    return check_feature_costs(costs, feature_shape, path)
