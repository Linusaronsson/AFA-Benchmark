import torch

from afabench.common.unmaskers.cube_nm_ar_unmasker import CubeNMARUnmasker


def test_selection_unmasks_context_then_direct_features() -> None:
    unmasker = CubeNMARUnmasker(n_contexts=3)
    feature_shape = torch.Size((8,))
    features = torch.arange(feature_shape.numel()).unsqueeze(0)
    feature_mask = torch.zeros_like(features, dtype=torch.bool)
    selection_mask = torch.zeros(
        (1, unmasker.get_n_selections(feature_shape)), dtype=torch.bool
    )

    context_mask = unmasker.unmask(
        masked_features=torch.zeros_like(features),
        feature_mask=feature_mask,
        features=features,
        afa_selection=torch.tensor([[0]]),
        selection_mask=selection_mask,
        label=None,
        feature_shape=feature_shape,
    )
    assert context_mask[0, :3].all()
    assert not context_mask[0, 3:].any()

    direct_mask = unmasker.unmask(
        masked_features=torch.zeros_like(features),
        feature_mask=feature_mask,
        features=features,
        afa_selection=torch.tensor([[1]]),
        selection_mask=selection_mask,
        label=None,
        feature_shape=feature_shape,
    )
    assert direct_mask[0, 3]
    assert not direct_mask[0, 4:].any()


def test_selection_costs_sum_context_and_keep_remaining_features_direct() -> (
    None
):
    unmasker = CubeNMARUnmasker(n_contexts=2)
    feature_costs = torch.tensor([0.5, 0.5, 1.0, 2.0, 4.0])

    selection_costs = unmasker.get_selection_costs(feature_costs)

    assert torch.allclose(selection_costs, torch.tensor([1.0, 1.0, 2.0, 4.0]))
