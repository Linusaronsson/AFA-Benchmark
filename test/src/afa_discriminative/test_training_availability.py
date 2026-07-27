import torch

from afabench.components.methods.discriminative.common.afa_methods import (
    _feature_marginal_selection_propensities,
)
from afabench.components.methods.discriminative.common.datasets import (
    prepare_datasets,
)
from afabench.components.unmaskers import (
    CubeNMUnmasker,
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
