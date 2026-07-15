from pathlib import Path
from typing import Self

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from afabench.components.initializers.config import InitializerConfig
from afabench.components.methods.discriminative.common import utils
from afabench.components.methods.discriminative.common.models import (
    MaskingPretrainer,
)
from afabench.components.unmaskers.config import UnmaskerConfig


class FakeDataset:
    feature_costs: torch.Tensor | None = None

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self._features: torch.Tensor = features
        self._labels: torch.Tensor = labels

    @property
    def feature_shape(self) -> torch.Size:
        return self._features.shape[1:]

    @property
    def label_shape(self) -> torch.Size:
        return self._labels.shape[1:]

    @classmethod
    def accepts_seed(cls) -> bool:
        return False

    def create_subset(self, indices: list[int]) -> Self:
        return self.__class__(self._features[indices], self._labels[indices])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._features[idx], self._labels[idx]

    def __len__(self) -> int:
        return len(self._features)

    def get_all_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._features, self._labels

    def save(self, path: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> Self:
        raise NotImplementedError


def test_training_prep_calculates_class_weights_for_image_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dataset = FakeDataset(
        features=torch.zeros((4, 1, 2, 2)),
        labels=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
    val_dataset = FakeDataset(
        features=torch.zeros((1, 1, 2, 2)),
        labels=torch.tensor([[0.0, 0.0, 1.0]]),
    )
    loaded_datasets = iter([(train_dataset, {}), (val_dataset, {})])

    monkeypatch.setattr(
        utils, "load_bundle", lambda _path: next(loaded_datasets)
    )
    monkeypatch.setattr(
        utils, "get_afa_initializer_from_config", lambda _cfg: object()
    )
    monkeypatch.setattr(
        utils, "get_afa_unmasker_from_config", lambda _cfg: object()
    )

    *_, class_weights = utils.afa_discriminative_training_prep(
        train_dataset_bundle_path=Path("train.bundle"),
        val_dataset_bundle_path=Path("val.bundle"),
        initializer_cfg=InitializerConfig(class_name="ignored", kwargs={}),
        unmasker_cfg=UnmaskerConfig(class_name="ignored", kwargs={}),
    )

    assert torch.allclose(class_weights, torch.tensor([0.2, 0.4, 0.4]))


def test_masking_pretrainer_flattens_image_features_for_tabular_model() -> (
    None
):
    features = torch.ones((4, 1, 2, 2))
    labels = torch.tensor([0, 1, 0, 1])
    loader = DataLoader(TensorDataset(features, labels), batch_size=2)
    model = nn.Sequential(nn.Flatten(), nn.Linear(8, 2))
    mask_layer = utils.MaskLayer(append=True)
    pretrainer = MaskingPretrainer(model, mask_layer)

    masked_features = mask_layer(features, torch.ones_like(features))

    assert masked_features.shape == (4, 8)

    pretrainer.fit(
        loader,
        loader,
        lr=0.01,
        nepochs=1,
        loss_fn=nn.CrossEntropyLoss(),
        verbose=False,
        min_mask=0.5,
        max_mask=0.5,
    )
