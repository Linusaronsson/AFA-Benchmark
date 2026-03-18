import torch

from afabench.common.initializers.cube_nm_ar_mar_initializer import (
    CubeNMARMARInitializer,
)


def test_initializer_pre_observes_context() -> None:
    initializer = CubeNMARMARInitializer(
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

    # Context features (first 5) should be pre-observed.
    assert observed[:, :5].all()
    # All other features should be unobserved.
    assert not observed[:, 5:].any()


def test_blocks_rescue_for_risky_samples() -> None:
    initializer = CubeNMARMARInitializer(
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

    assert forbidden.shape == (2, 22)
    assert not forbidden[0].any()
    assert forbidden[1, -1]
    assert not forbidden[1, :-1].any()


def test_training_forbidden_mask_marks_only_risky_rescue() -> None:
    initializer = CubeNMARMARInitializer(
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


def test_safe_context_never_blocks_rescue() -> None:
    initializer = CubeNMARMARInitializer(
        n_contexts=5,
        block_size=4,
        n_safe_contexts=2,
        risky_rescue_missing_probability=1.0,
    )
    feature_shape = torch.Size((26,))
    features = torch.zeros((4, feature_shape.numel()))
    features[0, 0] = 1.0  # safe context 0
    features[1, 1] = 1.0  # safe context 1
    features[2, 0] = 1.0  # safe context 0
    features[3, 1] = 1.0  # safe context 1

    observed = initializer.initialize(features, feature_shape=feature_shape)
    forbidden = initializer.get_training_forbidden_mask(observed)

    assert not forbidden.any()


def test_probability_zero_never_blocks() -> None:
    initializer = CubeNMARMARInitializer(
        n_contexts=5,
        block_size=4,
        n_safe_contexts=2,
        risky_rescue_missing_probability=0.0,
    )
    feature_shape = torch.Size((26,))
    features = torch.zeros((10, feature_shape.numel()))
    for i in range(10):
        features[i, 3] = 1.0  # all risky context

    observed = initializer.initialize(features, feature_shape=feature_shape)
    forbidden = initializer.get_training_forbidden_mask(observed)

    assert not forbidden.any()
