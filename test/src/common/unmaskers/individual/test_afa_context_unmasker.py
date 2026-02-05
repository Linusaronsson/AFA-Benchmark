import torch

from afabench.common.unmaskers.afa_context_unmasker import AFAContextUnmasker


def test_first_selection_unmasks_all_context_features() -> None:
    # Arrange
    feature_shape = torch.Size((33,))
    n_features = feature_shape.numel()
    n_contexts = 3
    unmasker = AFAContextUnmasker(n_contexts=n_contexts)
    features = torch.arange(n_features)  # feature values don't matter
    initial_feature_mask = torch.full((n_features,), False)
    initial_masked_features = features * initial_feature_mask.float()
    selection_mask = torch.full(
        (unmasker.get_n_selections(feature_shape),), False
    )

    # Act
    new_feature_mask = unmasker.unmask(
        masked_features=initial_masked_features,
        feature_mask=initial_feature_mask,
        features=features,
        afa_selection=torch.full((1, 1), 0),
        selection_mask=selection_mask,
        label=None,
        feature_shape=feature_shape,
    )

    # Assert
    expected_new_feature_mask = torch.full((n_features,), False)
    expected_new_feature_mask[:n_contexts] = True
    assert torch.allclose(new_feature_mask, expected_new_feature_mask)
