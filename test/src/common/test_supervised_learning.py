from collections.abc import Callable
from pathlib import Path
from typing import override

import pytest
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torch import nn

from afabench.core.bundle_system.bundle import save_bundle
from afabench.core.types import AFADataset
from afabench.datasets.datasets import CubeDataset
from afabench.training.config import SupervisedLearningConfig
from afabench.training.supervised_learning import (
    ensure_finite_module_state,
    supervised_learning,
)


def test_nonfinite_model_state_is_rejected() -> None:
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(float("nan"))

    with pytest.raises(FloatingPointError, match="weight"):
        ensure_finite_module_state(model)


class _TinyClassifier(LightningModule):
    def __init__(
        self, n_features: int, n_classes: int, *, nonfinite: bool
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)
        if nonfinite:
            with torch.no_grad():
                self.linear.weight.fill_(float("nan"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @override
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features, labels = batch
        return F.mse_loss(self(features), labels)

    @override
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        features, labels = batch
        self.log("val_loss", F.mse_loss(self(features), labels))

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _make_model_fn(
    *, nonfinite: bool
) -> Callable[[AFADataset], LightningModule]:
    def model_fn(dataset: AFADataset) -> LightningModule:
        return _TinyClassifier(
            n_features=dataset.feature_shape.numel(),
            n_classes=dataset.label_shape.numel(),
            nonfinite=nonfinite,
        )

    return model_fn


def _run_supervised_learning(
    tmp_path: Path, save_path: Path, *, nonfinite: bool
) -> None:
    train_bundle_path = tmp_path / "train.bundle"
    val_bundle_path = tmp_path / "val.bundle"
    save_bundle(CubeDataset(n_samples=10, seed=42), train_bundle_path, {})
    save_bundle(CubeDataset(n_samples=10, seed=43), val_bundle_path, {})

    cfg = SupervisedLearningConfig(
        batch_size=4,
        max_epochs=1,
        # Thresholds set above the single batch run below so that no
        # checkpoint is ever saved; the model's post-training state is
        # what gets checked for non-finite values and saved.
        checkpoint_earliest_batch=1000,
        early_stopping_min_batches=1000,
        early_stopping_patience=1,
        early_stopping_min_delta=0.0,
        val_check_interval=1,
        limit_train_batches=1,
        limit_val_batches=1,
    )

    supervised_learning(
        train_dataset_bundle_path=train_bundle_path,
        val_dataset_bundle_path=val_bundle_path,
        save_path=save_path,
        cfg=cfg,
        model_fn=_make_model_fn(nonfinite=nonfinite),
        metric_to_monitor="val_loss",
        monitor_mode="min",
        device="cpu",
    )


def test_supervised_learning_rejects_nonfinite_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    save_path = tmp_path / "model.bundle"

    with pytest.raises(FloatingPointError):
        _run_supervised_learning(tmp_path, save_path, nonfinite=True)

    assert not save_path.exists()


def test_supervised_learning_saves_finite_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    save_path = tmp_path / "model.bundle"

    _run_supervised_learning(tmp_path, save_path, nonfinite=False)

    assert save_path.exists()
    assert (save_path / "manifest.json").exists()
