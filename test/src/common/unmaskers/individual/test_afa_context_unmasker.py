from typing import Any

import pytest
import torch

from afabench.common.unmaskers.afa_context_unmasker import AFAContextUnmasker


@pytest.fixture
def unmasker_setup() -> dict[str, Any]:
    """Fixture providing common test setup."""
    n_contexts = 3
    feature_shape = torch.Size((33,))
    n_features = feature_shape.numel()
    unmasker = AFAContextUnmasker(n_contexts=n_contexts)

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


def test_selection_costs() -> None:
    n_contexts = 3
    unmasker = AFAContextUnmasker(n_contexts=n_contexts)

    selection_costs = unmasker.get_selection_costs(
        feature_costs=torch.tensor([0.2, 0.3, 0.1, 1, 2, 3])
    )

    expected_selection_costs = torch.tensor([0.6, 1, 2, 3])
    assert torch.allclose(selection_costs, expected_selection_costs)


def test_get_n_selections() -> None:
    n_contexts = 3
    unmasker = AFAContextUnmasker(n_contexts=n_contexts)

    n_selections = unmasker.get_n_selections(feature_shape=torch.Size((7,)))

    expected_n_selections = 5
    assert n_selections == expected_n_selections
