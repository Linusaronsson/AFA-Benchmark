"""PVAE completion of fixed training-data views."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from afabench.components.methods.rl.odin.models import ODINPretrainingModel
from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.core.utils import set_seed
from afabench.datasets.training_views import (
    TrainingDatasetView,
    pvae_restored_training_view,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from afabench.core.types import AFADataset
    from afabench.missing_values.config import RestoreTrainingViewsConfig

_STRATEGY_DISPLAY_NAMES = {
    "pvae_label_conditioned": "PVAE (label-conditioned)",
    "pvae_label_free": "PVAE (label-free)",
    "pvae_oracle": "PVAE (oracle)",
}


def _batches(n_rows: int, batch_size: int) -> Iterator[slice]:
    for start in range(0, n_rows, batch_size):
        yield slice(start, min(start + batch_size, n_rows))


def _load_view(path: Path) -> TrainingDatasetView:
    view, _ = load_bundle(path)
    if not isinstance(view, TrainingDatasetView):
        msg = f"Expected TrainingDatasetView at {path}."
        raise TypeError(msg)
    return view


def _load_pvae(path: Path, device: torch.device) -> ODINPretrainingModel:
    bundle, _ = load_bundle(path, device=device)
    if not isinstance(bundle, TorchModelBundle):
        msg = f"Expected TorchModelBundle at {path}."
        raise TypeError(msg)
    model = bundle.model
    if not isinstance(model, ODINPretrainingModel):
        msg = f"Expected an ODINPretrainingModel at {path}."
        raise TypeError(msg)
    model = model.to(device)
    model.eval()
    return model


def restore_view_with_pvae(
    view: TrainingDatasetView,
    model: ODINPretrainingModel,
    *,
    strategy: str,
    batch_size: int,
) -> TrainingDatasetView:
    """Draw one joint reconstruction per row and preserve factual cells."""
    if strategy not in _STRATEGY_DISPLAY_NAMES:
        msg = f"Unsupported PVAE restoration strategy: {strategy}"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = "batch_size must be positive."
        raise ValueError(msg)
    use_label = strategy != "pvae_label_free"
    device = model.device
    reconstructed: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_slice in _batches(len(view), batch_size):
            features = view.features[batch_slice].to(device)
            availability = view.source_availability[batch_slice].to(device)
            labels = view.labels[batch_slice].to(device) if use_label else None
            _, estimate = model.masked_reconstruction(
                masked_features=features,
                feature_mask=availability,
                n_classes=view.label_shape.numel(),
                label=labels,
            )
            reconstructed.append(estimate.cpu())
    return pvae_restored_training_view(
        view,
        torch.cat(reconstructed),
        strategy=strategy,
    )


def _reference_rmse(
    restored: TrainingDatasetView,
    reference_path: Path | None,
) -> float | None:
    if reference_path is None:
        return None
    reference, _ = load_bundle(reference_path)
    reference = cast("AFADataset", cast("object", reference))
    reference_features, _ = reference.get_all_data()
    missing = ~restored.source_availability
    if not missing.any():
        return 0.0
    squared_error = (
        restored.features[missing] - reference_features[missing]
    ).square()
    return float(squared_error.mean().sqrt().item())


def restore_and_save_training_views(cfg: RestoreTrainingViewsConfig) -> None:
    """Restore and bundle a train/validation pair with shared PVAE weights."""
    if cfg.strategy not in _STRATEGY_DISPLAY_NAMES:
        msg = f"Unsupported PVAE restoration strategy: {cfg.strategy}"
        raise ValueError(msg)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = _load_pvae(cfg.pvae_bundle_path, device)
    train_view = restore_view_with_pvae(
        _load_view(cfg.train_view_bundle_path),
        model,
        strategy=cfg.strategy,
        batch_size=cfg.batch_size,
    )
    val_view = restore_view_with_pvae(
        _load_view(cfg.val_view_bundle_path),
        model,
        strategy=cfg.strategy,
        batch_size=cfg.batch_size,
    )
    common_metadata = {
        "strategy": cfg.strategy,
        "strategy_display_name": _STRATEGY_DISPLAY_NAMES[cfg.strategy],
        "pvae_bundle_path": str(cfg.pvae_bundle_path),
        "restoration_seed": cfg.seed,
        "source_dataset_class_name": train_view.source_dataset_class_name,
    }
    save_bundle(
        train_view,
        cfg.train_save_path,
        common_metadata
        | {
            "split": "train",
            "imputation_rmse": _reference_rmse(
                train_view,
                cfg.reference_train_dataset_bundle_path,
            ),
        },
    )
    save_bundle(
        val_view,
        cfg.val_save_path,
        common_metadata
        | {
            "split": "val",
            "imputation_rmse": _reference_rmse(
                val_view,
                cfg.reference_val_dataset_bundle_path,
            ),
        },
    )
