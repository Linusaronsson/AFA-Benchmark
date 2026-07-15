"""Materialization of immutable train/validation views under missingness."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, cast

from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.datasets.training_views import (
    TrainingDatasetView,
    mean_filled_training_views,
    restricted_training_view,
    true_value_training_view,
    zero_filled_training_view,
)
from afabench.missing_values.masking import FittedMissingnessMechanism

if TYPE_CHECKING:
    from pathlib import Path

    from afabench.core.types import AFADataset
    from afabench.missing_values.config import MaterializeTrainingViewsConfig


def _load_dataset(path: Path) -> tuple[AFADataset, dict[str, object]]:
    dataset, manifest = load_bundle(path)
    return cast("AFADataset", cast("object", dataset)), manifest


def materialize_training_views(
    cfg: MaterializeTrainingViewsConfig,
) -> tuple[TrainingDatasetView, TrainingDatasetView]:
    """Fit on train, then deterministically mask both train and validation."""
    train_dataset, _ = _load_dataset(cfg.train_dataset_bundle_path)
    val_dataset, _ = _load_dataset(cfg.val_dataset_bundle_path)
    train_features, _ = train_dataset.get_all_data()
    val_features, _ = val_dataset.get_all_data()

    mechanism = FittedMissingnessMechanism.fit(
        train_features,
        cfg.missingness,
        seed=cfg.seed,
    )
    train_available = mechanism.sample_observed_mask(
        train_features,
        seed=cfg.seed + 1,
    )
    val_available = mechanism.sample_observed_mask(
        val_features,
        seed=cfg.seed + 2,
    )
    train_restricted = restricted_training_view(
        train_dataset,
        train_available,
    )
    val_restricted = restricted_training_view(val_dataset, val_available)

    if cfg.strategy == "restricted":
        return train_restricted, val_restricted
    if cfg.strategy == "mean_fill":
        return mean_filled_training_views(train_restricted, val_restricted)
    if cfg.strategy == "true_completion":
        return (
            true_value_training_view(train_dataset, train_restricted),
            true_value_training_view(val_dataset, val_restricted),
        )
    if cfg.strategy == "zero_fill":
        return (
            zero_filled_training_view(train_restricted),
            zero_filled_training_view(val_restricted),
        )
    msg = f"Unsupported training-view strategy: {cfg.strategy}"
    raise ValueError(msg)


def save_materialized_training_views(
    cfg: MaterializeTrainingViewsConfig,
) -> None:
    """Materialize and bundle a train/validation view pair."""
    train_view, val_view = materialize_training_views(cfg)
    common_metadata = {
        "missingness": asdict(cfg.missingness),
        "missingness_seed": cfg.seed,
        "strategy": cfg.strategy,
        "source_dataset_class_name": train_view.source_dataset_class_name,
    }
    save_bundle(
        train_view,
        cfg.train_save_path,
        common_metadata
        | {
            "split": "train",
            "source_dataset_bundle_path": str(cfg.train_dataset_bundle_path),
            "source_observed_fraction": float(
                train_view.source_availability.float().mean().item()
            ),
        },
    )
    save_bundle(
        val_view,
        cfg.val_save_path,
        common_metadata
        | {
            "split": "val",
            "source_dataset_bundle_path": str(cfg.val_dataset_bundle_path),
            "source_observed_fraction": float(
                val_view.source_availability.float().mean().item()
            ),
        },
    )
