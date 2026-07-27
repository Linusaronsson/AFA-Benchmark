import pytest
import torch

from afabench.components.unmaskers import GroupedFeatureUnmasker


@pytest.fixture
def unmasker() -> GroupedFeatureUnmasker:
    return GroupedFeatureUnmasker(group_ids=[5, 5, 2, 9, 9, 9])


def test_grouped_costs_sum_apportioned_feature_costs(
    unmasker: GroupedFeatureUnmasker,
) -> None:
    costs = unmasker.get_selection_costs(
        torch.tensor([1.0, 2.0, 4.0, 2.0, 3.0, 5.0])
    )

    assert torch.equal(costs, torch.tensor([4.0, 3.0, 10.0]))


def test_group_availability_requires_every_processed_column(
    unmasker: GroupedFeatureUnmasker,
) -> None:
    availability = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, False, True, True, True, True],
        ]
    )

    selections = unmasker.feature_availability_to_selection_availability(
        availability
    )

    assert torch.equal(
        selections,
        torch.tensor([[True, True, True], [True, False, True]]),
    )


def test_unmask_reveals_only_selected_group(
    unmasker: GroupedFeatureUnmasker,
) -> None:
    features = torch.arange(12).reshape(2, 6).float()
    feature_mask = torch.zeros_like(features, dtype=torch.bool)

    result = unmasker.unmask(
        masked_features=torch.zeros_like(features),
        feature_mask=feature_mask,
        features=features,
        afa_selection=torch.tensor([[1], [2]]),
        selection_mask=torch.zeros((2, 3), dtype=torch.bool),
        feature_shape=torch.Size([6]),
    )

    assert torch.equal(
        result,
        torch.tensor(
            [
                [True, True, False, False, False, False],
                [False, False, False, True, True, True],
            ]
        ),
    )
