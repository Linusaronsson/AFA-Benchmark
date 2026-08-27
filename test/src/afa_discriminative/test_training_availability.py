from typing import override

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from afabench.components.initializers.zero_initializer import ZeroInitializer
from afabench.components.methods.discriminative.common.afa_methods import (
    GreedyDynamicSelection,
    _feature_marginal_selection_propensities,
    _initial_training_masks,
)
from afabench.components.methods.discriminative.common.datasets import (
    prepare_datasets,
)
from afabench.components.methods.discriminative.common.models import (
    MaskingPretrainer,
)
from afabench.components.methods.discriminative.common.utils import MaskLayer
from afabench.components.unmaskers import (
    CubeNMUnmasker,
    DirectUnmasker,
    GroupedFeatureUnmasker,
)
from afabench.datasets.datasets import CubeNMDataset
from afabench.datasets.training_views import restricted_training_view


def test_discriminative_loader_exposes_both_availability_masks() -> None:
    dataset = CubeNMDataset(n_samples=20, seed=1, n_contexts=2)
    source_availability = torch.ones_like(dataset.features, dtype=torch.bool)
    source_availability[::2, 3] = False
    view = restricted_training_view(dataset, source_availability)

    train_loader, _val_loader, _d_in, _d_out = prepare_datasets(
        view,
        view,
        batch_size=4,
    )
    batch = next(iter(train_loader))

    assert len(batch) == 4
    features, _labels, factual_support, selectable_support = batch
    assert (features[~factual_support] == 0).all()
    assert torch.equal(factual_support, selectable_support)


def test_cube_context_propensity_uses_atomic_selection_availability() -> None:
    source_availability = torch.ones((2, 52), dtype=torch.bool)
    source_availability[0, :2] = False
    source_availability[0, 2] = False
    unmasker = CubeNMUnmasker(n_contexts=2)

    propensities = _feature_marginal_selection_propensities(
        source_availability,
        unmasker,
    )

    assert propensities.shape == (51,)
    assert torch.isclose(propensities[0], torch.tensor(0.5))
    assert torch.isclose(propensities[1], torch.tensor(0.5))
    assert torch.isclose(propensities[2], torch.tensor(1.0))


def test_generic_group_propensities_match_selection_space() -> None:
    source_availability = torch.tensor(
        [
            [False, False, True],
            [True, True, False],
            [True, True, True],
        ]
    )
    unmasker = GroupedFeatureUnmasker(group_ids=[0, 0, 1])

    propensities = _feature_marginal_selection_propensities(
        source_availability,
        unmasker,
    )

    assert torch.allclose(propensities, torch.tensor([2 / 3, 2 / 3]))


class _RecordingMaskLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.masks: list[torch.Tensor] = []

    @override
    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        self.masks.append(mask.detach().clone())
        return features * mask


def test_gdfs_pretraining_uses_effective_training_support() -> None:
    features = torch.tensor([[1.0, 90.0], [70.0, 2.0]])
    labels = torch.tensor([0, 1])
    source_availability = torch.tensor([[True, False], [False, True]])
    training_availability = torch.tensor([[True, True], [False, True]])
    layer = _RecordingMaskLayer()
    pretrainer = MaskingPretrainer(torch.nn.Linear(2, 2), layer)
    loader = [(features, labels, source_availability, training_availability)]

    pretrainer.fit(
        loader,
        loader,
        lr=1e-3,
        nepochs=1,
        loss_fn=torch.nn.CrossEntropyLoss(),
        min_mask=0.0,
        max_mask=0.0,
        verbose=False,
    )

    assert layer.masks
    assert all(
        torch.equal(mask.bool(), training_availability) for mask in layer.masks
    )
    assert any(
        (mask.bool() & ~source_availability).any() for mask in layer.masks
    )


def test_gdfs_starts_unavailable_selections_exhausted() -> None:
    availability = torch.tensor([[True, False, True], [False, False, False]])

    _, selection_mask = _initial_training_masks(
        features=torch.zeros((2, 3)),
        labels=torch.tensor([0, 1]),
        feature_shape=torch.Size((3,)),
        initializer=ZeroInitializer(),
        selection_availability=availability,
    )

    assert torch.equal(selection_mask.bool(), ~availability)


def test_gdfs_selector_gradient_handles_exhausted_instances() -> None:
    torch.manual_seed(4)
    features = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 0, 1])
    availability = torch.tensor(
        [[False, False], [True, True], [True, False], [False, True]]
    )
    loader = DataLoader(
        TensorDataset(features, labels, availability, availability),
        batch_size=4,
    )
    method = GreedyDynamicSelection(
        selector=nn.Linear(4, 2),
        predictor=nn.Linear(4, 2),
        mask_layer=MaskLayer(append=True),
        initializer=ZeroInitializer(),
        unmasker=DirectUnmasker(),
    )

    method.fit(
        loader,
        loader,
        lr=1e-3,
        nepochs=1,
        max_features=2,
        loss_fn=nn.CrossEntropyLoss(),
        temp_steps=1,
        verbose=False,
    )

    assert all(
        torch.isfinite(parameter).all() for parameter in method.parameters()
    )
