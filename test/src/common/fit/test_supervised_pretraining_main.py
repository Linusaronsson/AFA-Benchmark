from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pytest
import torch
import torch.nn.functional as F
from jaxtyping import Float
from lightning import LightningModule
from torch import Tensor, nn

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import (
    bundle_provenance,
    load_bundle,
    save_bundle,
)
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.core.types import AFADataset
from afabench.datasets.datasets import CubeDataset
from afabench.fit.config import (
    SupervisedLearningConfig,
    SupervisedPretrainingConfig,
)
from afabench.fit.inputs import FitInputs
from afabench.fit.supervised_learning import supervised_pretraining_main
from afabench.testing.provenance import placeholder_provenance

type Features = Float[Tensor, "batch n_features"]
type Labels = Float[Tensor, "batch n_classes"]
type Logits = Float[Tensor, "batch n_classes"]
type Loss = Float[Tensor, ""]


@dataclass(frozen=True, kw_only=True)
class _TinyPretrainConfig(SupervisedPretrainingConfig):
    hidden_size: int


class _TinyModel(LightningModule):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_classes)

    def forward(self, x: Features) -> Logits:
        return self.linear(x)

    @override
    def training_step(
        self, batch: tuple[Features, Labels], batch_idx: int
    ) -> Loss:
        features, labels = batch
        return F.mse_loss(self(features), labels)

    @override
    def validation_step(
        self, batch: tuple[Features, Labels], batch_idx: int
    ) -> None:
        features, labels = batch
        self.log(
            "val_loss_many_observations", F.mse_loss(self(features), labels)
        )

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


@pytest.fixture
def pretrain_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _TinyPretrainConfig:
    # Lightning writes its logs relative to the working directory
    monkeypatch.chdir(tmp_path)
    for split, n_samples in (("train", 10), ("val", 6)):
        save_bundle(
            CubeDataset(n_samples=n_samples, seed=n_samples),
            tmp_path / f"{split}.bundle",
            metadata={},
            provenance=placeholder_provenance(
                dataset_key="cube", dataset_realization_index=0, split=split
            ),
        )
    save_bundle(
        TorchModelBundle(nn.Linear(2, 2)),
        tmp_path / "classifier.bundle",
        metadata={},
        provenance=placeholder_provenance("classifier_training"),
    )
    return _TinyPretrainConfig(
        train_dataset_bundle_path=str(tmp_path / "train.bundle"),
        val_dataset_bundle_path=str(tmp_path / "val.bundle"),
        classifier_bundle_path=str(tmp_path / "classifier.bundle"),
        save_path=str(tmp_path / "model.bundle"),
        initializer=InitializerConfig(class_name="ColdInitializer", kwargs={}),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        device="cpu",
        seed=0,
        smoke_test=True,
        supervised_learning=SupervisedLearningConfig(
            batch_size=4,
            max_epochs=100,
            # Above the batches run, so that the metric is never monitored
            checkpoint_earliest_batch=1000,
            early_stopping_min_batches=1000,
            early_stopping_patience=1,
            early_stopping_min_delta=0.0,
            val_check_interval=1,
            limit_train_batches=None,
            limit_val_batches=None,
        ),
        hidden_size=3,
    )


def _tiny_model_fn(
    configs_seen: list[_TinyPretrainConfig],
) -> Callable[
    [_TinyPretrainConfig, FitInputs], Callable[[AFADataset], LightningModule]
]:
    def build_model_fn(
        cfg: _TinyPretrainConfig, _inputs: FitInputs
    ) -> Callable[[AFADataset], LightningModule]:
        configs_seen.append(cfg)
        return lambda dataset: _TinyModel(
            dataset.feature_shape.numel(), dataset.label_shape.numel()
        )

    return build_model_fn


def test_supervised_pretraining_main_saves_the_model_at_save_path(
    pretrain_config: _TinyPretrainConfig,
) -> None:
    supervised_pretraining_main(
        pretrain_config, tags=["tiny"], model_fn=_tiny_model_fn([])
    )

    loaded = load_bundle(Path(pretrain_config.save_path))
    assert isinstance(loaded.obj, TorchModelBundle)
    assert isinstance(loaded.obj.model, _TinyModel)
    assert loaded.manifest.metadata["method_config"]["hidden_size"] == 3
    record = bundle_provenance(Path(pretrain_config.save_path))
    assert record is not None
    assert record.stage == "pretraining"


def test_supervised_pretraining_main_builds_the_model_from_the_limited_config(
    pretrain_config: _TinyPretrainConfig,
) -> None:
    configs_seen: list[_TinyPretrainConfig] = []

    supervised_pretraining_main(
        pretrain_config, tags=["tiny"], model_fn=_tiny_model_fn(configs_seen)
    )

    [cfg] = configs_seen
    assert cfg.supervised_learning.max_epochs == 1
