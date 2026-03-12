import torch

from afabench.common.initializers.cube_nm_ar_initializer import (
    CubeNMARInitializer,
)
from afabench.common.initializers.random_initializer import RandomInitializer
from scripts.train.aaco import (
    _apply_train_missingness,
    _derive_train_support_masks,
)


def test_random_initializer_keeps_full_training_support_for_aaco() -> None:
    initializer = RandomInitializer(num_initial_features=0)
    features = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    observed_mask, train_support_mask = _derive_train_support_masks(
        initializer,
        seed=0,
        features=features,
        feature_shape=torch.Size((6,)),
    )

    assert not observed_mask.any()
    assert train_support_mask.all()


def test_cube_initializer_distinguishes_start_mask_from_training_support() -> (
    None
):
    initializer = CubeNMARInitializer(
        n_contexts=5,
        block_size=4,
        n_safe_contexts=2,
        risky_rescue_missing_probability=1.0,
    )
    feature_shape = torch.Size((26,))
    features = torch.zeros((2, feature_shape.numel()))
    features[0, 0] = 1.0
    features[1, 3] = 1.0
    features[:, -1] = 4.0 / 7.0

    observed_mask, train_support_mask = _derive_train_support_masks(
        initializer,
        seed=0,
        features=features,
        feature_shape=feature_shape,
    )

    assert not observed_mask[0, :5].any()
    assert train_support_mask[0, :5].all()
    assert train_support_mask[0, -1]

    assert not observed_mask[1].any()
    assert not train_support_mask[1, -1]
    assert train_support_mask[1, :-1].all()


def test_aaco_missingness_uses_training_support_not_start_mask() -> None:
    initializer = CubeNMARInitializer(
        n_contexts=5,
        block_size=4,
        n_safe_contexts=2,
        risky_rescue_missing_probability=1.0,
    )
    feature_shape = torch.Size((26,))
    features = torch.zeros((2, feature_shape.numel()))
    features[0, 0] = 1.0
    features[1, 3] = 1.0
    features[:, -1] = 4.0 / 7.0

    _observed_mask, train_support_mask = _derive_train_support_masks(
        initializer,
        seed=0,
        features=features,
        feature_shape=feature_shape,
    )

    zero_filled, zero_fill_mask = _apply_train_missingness(
        x_train=features,
        train_support_mask=train_support_mask,
        mode="zero_fill",
        hide_val=0.0,
    )
    mask_aware_filled, mask_aware_mask = _apply_train_missingness(
        x_train=features,
        train_support_mask=train_support_mask,
        mode="mask_aware",
        hide_val=0.0,
    )

    assert zero_fill_mask is None
    assert bool(zero_filled[0, 0].item())
    assert bool(zero_filled[0, -1].item())
    assert not bool(zero_filled[1, -1].item())

    assert torch.equal(mask_aware_filled, zero_filled)
    assert mask_aware_mask is not None
    assert torch.equal(mask_aware_mask, train_support_mask)
