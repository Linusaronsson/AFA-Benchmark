import torch

from afabench.common.initializers.cube_nm_ar_initializer import (
    CubeNMARInitializer,
)


def test_initializer_starts_cold_and_blocks_rescue_for_risky_samples() -> None:
    initializer = CubeNMARInitializer(
        n_contexts=5,
        block_size=4,
        n_safe_contexts=2,
        risky_rescue_missing_probability=1.0,
    )
    feature_shape = torch.Size((26,))

    features = torch.zeros((2, feature_shape.numel()))
    features[0, 0] = 1.0  # safe context
    features[1, 3] = 1.0  # risky context

    observed = initializer.initialize(features, feature_shape=feature_shape)
    forbidden = initializer.get_forbidden_selection_mask(
        observed, feature_shape
    )

    assert not observed.any()
    assert forbidden.shape == (2, 22)
    assert not forbidden[0].any()
    assert forbidden[1, -1]
    assert not forbidden[1, :-1].any()


def test_training_forbidden_mask_marks_only_risky_rescue_feature() -> None:
    initializer = CubeNMARInitializer(
        n_contexts=5,
        block_size=4,
        n_safe_contexts=2,
        risky_rescue_missing_probability=1.0,
    )
    feature_shape = torch.Size((26,))
    features = torch.zeros((2, feature_shape.numel()))
    features[0, 1] = 1.0  # safe context
    features[1, 4] = 1.0  # risky context

    observed = initializer.initialize(features, feature_shape=feature_shape)
    forbidden = initializer.get_training_forbidden_mask(observed)

    assert not forbidden[0].any()
    assert forbidden[1, -1]
    assert not forbidden[1, :-1].any()
