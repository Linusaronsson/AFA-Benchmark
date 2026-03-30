from __future__ import annotations

from typing import final

import torch

from afabench.afa_discriminative.afa_methods import (
    _build_initial_feature_and_selection_masks,
)
from afabench.afa_discriminative.datasets import prepare_datasets
from afabench.common.unmaskers.cube_nm_ar_unmasker import CubeNMARUnmasker


@final
class _ToyDataset:
    def __init__(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self._features = features
        self._labels = labels

    @property
    def feature_shape(self) -> torch.Size:
        return torch.Size((self._features.shape[1],))

    @property
    def label_shape(self) -> torch.Size:
        return torch.Size((self._labels.shape[1],))

    def get_all_data(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._features, self._labels

    def __len__(self) -> int:
        return int(self._features.shape[0])


def test_prepare_datasets_preserves_observed_and_forbidden_masks() -> None:
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    observed_mask = torch.tensor(
        [[True, False], [False, True]],
        dtype=torch.bool,
    )
    forbidden_mask = ~observed_mask
    dataset = _ToyDataset(features, labels)

    _train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset=dataset,
        val_dataset=dataset,
        batch_size=2,
        train_observed_mask=observed_mask,
        val_observed_mask=observed_mask,
        train_forbidden_mask=forbidden_mask,
        val_forbidden_mask=forbidden_mask,
    )

    batch_features, batch_labels, batch_observed, batch_forbidden = next(
        iter(val_loader)
    )

    assert d_in == 2
    assert d_out == 2
    assert torch.equal(batch_features, features)
    assert torch.equal(batch_labels, torch.tensor([0, 1]))
    assert torch.equal(batch_observed, observed_mask)
    assert torch.equal(batch_forbidden, forbidden_mask)


def test_initial_masks_mark_preobserved_grouped_context_as_unavailable() -> (
    None
):
    observed_mask = torch.zeros((2, 26), dtype=torch.bool)
    observed_mask[0, :5] = True
    observed_mask[1, :4] = True
    observed_mask[1, 7] = True

    forbidden_mask = torch.zeros_like(observed_mask)
    forbidden_mask[:, -1] = True

    feature_mask, selection_mask = _build_initial_feature_and_selection_masks(
        observed_mask,
        forbidden_mask,
        n_selections=22,
        feature_shape=torch.Size((26,)),
        unmasker=CubeNMARUnmasker(n_contexts=5),
        dtype=torch.float32,
    )

    assert torch.equal(feature_mask, observed_mask.float())
    assert feature_mask.dtype == torch.float32

    assert selection_mask.shape == (2, 22)
    assert selection_mask[0, 0]
    assert selection_mask[0, -1]
    assert selection_mask[0].sum().item() == 2

    assert not selection_mask[1, 0]
    assert selection_mask[1, 3]
    assert selection_mask[1, -1]
    assert selection_mask[1].sum().item() == 2
