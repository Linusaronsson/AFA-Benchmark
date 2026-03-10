import torch

from afabench.common.initializers.cube_nm_ar_initializer import (
    CubeNMARInitializer,
)


def test_initializer_reveals_hint_pattern_and_blocks_risky_relevant_block() -> (
    None
):
    initializer = CubeNMARInitializer(
        n_contexts=5,
        n_hint_features=5,
        block_size=4,
        n_safe_contexts=2,
        n_admin_features=2,
    )
    feature_shape = torch.Size((33,))

    features = torch.zeros((2, feature_shape.numel()))
    # Context one-hot.
    features[0, 0] = 1.0  # safe context
    features[1, 3] = 1.0  # risky context
    # Hint features are constant-valued in the dataset.
    features[:, 5:10] = 1.0
    # Hidden admin features encode hint regime and blockage; rescue is irrelevant.
    features[0, 10] = 1.0  # ambiguous hint regime
    features[0, 11] = 0.0  # not blocked
    features[1, 10] = 2.0  # no hint
    features[1, 11] = 1.0  # blocked risky block
    features[:, -1] = 4.0 / 7.0

    observed = initializer.initialize(features, feature_shape=feature_shape)
    forbidden = initializer.get_forbidden_selection_mask(
        observed, feature_shape
    )

    # Sample 0: ambiguous regime reveals the true hint and one decoy hint.
    assert observed[0, 5]
    assert observed[0, 6]
    assert observed[0, 5:10].sum() == 2

    # Sample 1: no hint revealed.
    assert observed[1, 5:10].sum() == 0

    # Risky blocked sample forbids its relevant block (context 3 -> block selections 13:17).
    assert forbidden.shape == (2, 22)
    assert forbidden[1, 13:17].all()
    assert not forbidden[0].any()


def test_training_forbidden_mask_preserves_observed_hints_and_blocks_risky_block() -> (
    None
):
    initializer = CubeNMARInitializer(
        n_contexts=5,
        n_hint_features=5,
        block_size=4,
        n_safe_contexts=2,
        n_admin_features=2,
    )
    feature_shape = torch.Size((33,))
    features = torch.zeros((2, feature_shape.numel()))
    features[0, 0] = 1.0  # safe context
    features[1, 3] = 1.0  # risky context
    features[:, 5:10] = 1.0
    features[0, 10] = 1.0  # ambiguous hint regime
    features[0, 11] = 0.0
    features[1, 10] = 2.0  # no hint
    features[1, 11] = 1.0  # blocked risky block

    observed = initializer.initialize(features, feature_shape=feature_shape)
    forbidden = initializer.get_training_forbidden_mask(observed)

    # Sample 0 keeps its two revealed hints visible and forbids the rest.
    assert not forbidden[0, 5]
    assert not forbidden[0, 6]
    assert forbidden[0, 7:10].all()

    # Admin controls are always forbidden.
    assert forbidden[:, 10:12].all()

    # Risky blocked sample forbids its relevant block in feature space.
    assert forbidden[1, 24:28].all()
    assert not forbidden[1, -1]
    assert not forbidden[:, :5].any()
