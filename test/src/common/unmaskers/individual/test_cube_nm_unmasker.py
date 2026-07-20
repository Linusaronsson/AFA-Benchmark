from typing import Any

import pytest
import torch

from afabench.components.unmaskers.cube_nm_unmasker import CubeNMUnmasker


@pytest.fixture
def unmasker_setup() -> dict[str, Any]:
    """Fixture providing common test setup."""
    n_contexts = 3
    feature_shape = torch.Size((33,))
    n_features = feature_shape.numel()
    unmasker = CubeNMUnmasker(n_contexts=n_contexts)

    features = torch.arange(n_features).unsqueeze(0)
    initial_feature_mask = torch.full((n_features,), False).unsqueeze(0)
    initial_masked_features = features * initial_feature_mask.float()
    selection_mask = torch.full(
        (unmasker.get_n_selections(feature_shape),), False
    ).unsqueeze(0)

    return {
        "n_contexts": n_contexts,
        "feature_shape": feature_shape,
        "unmasker": unmasker,
        "features": features,
        "initial_feature_mask": initial_feature_mask,
        "initial_masked_features": initial_masked_features,
        "selection_mask": selection_mask,
    }


def unmask_and_assert(
    setup: dict[str, Any],
    afa_selection_idx: int,
    expected_indices: int | slice,
) -> None:
    """Execute unmask and assert results."""
    afa_selection = torch.full((1, 1), afa_selection_idx)

    new_feature_mask = setup["unmasker"].unmask(
        masked_features=setup["initial_masked_features"],
        feature_mask=setup["initial_feature_mask"],
        features=setup["features"],
        afa_selection=afa_selection,
        selection_mask=setup["selection_mask"],
        label=None,
        feature_shape=setup["feature_shape"],
    )

    expected_new_feature_mask = torch.full(
        (setup["feature_shape"].numel(),), False
    )
    expected_new_feature_mask[expected_indices] = True
    assert torch.allclose(new_feature_mask, expected_new_feature_mask)


@pytest.mark.parametrize(
    ("afa_selection_idx", "expected_indices"),
    [
        (0, slice(0, 3)),  # First selection: all context features
        (1, 3),  # Second selection: first normal feature
        (4, 6),  # Fifth selection: fourth normal feature (3 + 3)
    ],
)
def test_selection_unmasks_correct_features(
    unmasker_setup: dict[str, Any],
    afa_selection_idx: int,
    expected_indices: int | slice,
) -> None:
    unmask_and_assert(unmasker_setup, afa_selection_idx, expected_indices)


def test_mixed_batch_unmasks_each_sample_independently() -> None:
    """
    The context branch and the individual branch have to coexist in one batch.

    The per-sample loop this replaced could not get this wrong; the vectorized
    form can, by letting one sample's branch write into another's row.
    """
    unmasker = CubeNMUnmasker(n_contexts=3)
    n_features = 8
    features = torch.arange(n_features).expand(3, n_features).float()
    feature_mask = torch.zeros((3, n_features), dtype=torch.bool)
    # A bit already set must survive, so this cannot pass by rebuilding masks.
    feature_mask[2, 7] = True
    # Sample 0 takes the context group, samples 1 and 2 take single features.
    afa_selection = torch.tensor([[0], [1], [4]])

    new_feature_mask = unmasker.unmask(
        masked_features=features * feature_mask.float(),
        feature_mask=feature_mask,
        features=features,
        afa_selection=afa_selection,
        selection_mask=torch.zeros((3, 6), dtype=torch.bool),
        label=None,
        feature_shape=torch.Size((n_features,)),
    )

    expected = torch.zeros((3, n_features), dtype=torch.bool)
    expected[0, :3] = True  # context group
    expected[1, 3] = True  # first individual feature
    expected[2, 6] = True  # fourth individual feature
    expected[2, 7] = True  # and the bit that was already set
    assert torch.equal(new_feature_mask, expected)


def test_empty_batch_is_a_no_op() -> None:
    """Callers gather only the rows with an available selection, sometimes none."""
    unmasker = CubeNMUnmasker(n_contexts=3)
    empty_features = torch.zeros((0, 8))
    empty_mask = torch.zeros((0, 8), dtype=torch.bool)

    new_feature_mask = unmasker.unmask(
        masked_features=empty_features,
        feature_mask=empty_mask,
        features=empty_features,
        afa_selection=torch.zeros((0, 1), dtype=torch.long),
        selection_mask=torch.zeros((0, 6), dtype=torch.bool),
        label=None,
        feature_shape=torch.Size((8,)),
    )

    assert new_feature_mask.shape == (0, 8)


def test_selection_costs() -> None:
    n_contexts = 3
    unmasker = CubeNMUnmasker(n_contexts=n_contexts)

    selection_costs = unmasker.get_selection_costs(
        feature_costs=torch.tensor([0.2, 0.3, 0.1, 1, 2, 3])
    )

    expected_selection_costs = torch.tensor([0.6, 1, 2, 3])
    assert torch.allclose(selection_costs, expected_selection_costs)


def test_get_n_selections() -> None:
    n_contexts = 3
    unmasker = CubeNMUnmasker(n_contexts=n_contexts)

    n_selections = unmasker.get_n_selections(feature_shape=torch.Size((7,)))

    expected_n_selections = 5
    assert n_selections == expected_n_selections


def test_feature_availability_requires_the_whole_context_group() -> None:
    unmasker = CubeNMUnmasker(n_contexts=3)
    feature_availability = torch.tensor(
        [
            [True, True, True, True, False],
            [True, False, True, True, True],
        ]
    )

    selection_availability = (
        unmasker.feature_availability_to_selection_availability(
            feature_availability
        )
    )

    assert torch.equal(
        selection_availability,
        torch.tensor([[True, True, False], [False, True, True]]),
    )
