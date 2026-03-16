from __future__ import annotations

import torch

from afabench.afa_oracle.aaco_core import AACOOracle
from afabench.common.initializers.cube_nm_ar_initializer import (
    CubeNMARInitializer,
)
from afabench.common.initializers.random_initializer import RandomInitializer
from scripts.train.aaco import (
    _apply_train_missingness,
    _derive_train_support_masks,
)


class _MaskAwareToyClassifier:
    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del feature_shape
        positive_class = x[:, 0] >= 0
        has_second_feature = mask[:, 1] > 0.5
        probs = torch.empty((x.shape[0], 2), dtype=torch.float32)

        probs[positive_class & has_second_feature] = torch.tensor([0.9, 0.1])
        probs[positive_class & ~has_second_feature] = torch.tensor([0.6, 0.4])
        probs[~positive_class & has_second_feature] = torch.tensor([0.1, 0.9])
        probs[~positive_class & ~has_second_feature] = torch.tensor([0.4, 0.6])
        return probs


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


def test_aaco_dr_reweights_supported_neighbors_toward_full_candidate_loss() -> (
    None
):
    x_train = torch.tensor(
        [[1.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    y_train = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    observed_mask = torch.tensor(
        [[True, True], [True, False]],
        dtype=torch.bool,
    )
    candidate_mask = torch.tensor([[True, True]], dtype=torch.bool)
    idx_nn = torch.tensor([0, 1], dtype=torch.long)

    mask_aware = AACOOracle(
        k_neighbors=2,
        acquisition_cost=0.0,
        missingness_objective="mask_aware",
    )
    mask_aware.set_classifier(_MaskAwareToyClassifier())
    mask_aware.fit(x_train, y_train, observed_mask=observed_mask)
    mask_aware_loss = mask_aware._compute_expected_losses(  # noqa: SLF001
        candidate_feature_masks=candidate_mask,
        idx_nn=idx_nn,
        feature_count=2,
    )

    dr_oracle = AACOOracle(
        k_neighbors=2,
        acquisition_cost=0.0,
        missingness_objective="doubly_robust",
        dr_max_weight=None,
    )
    dr_oracle.set_classifier(_MaskAwareToyClassifier())
    dr_oracle.fit(x_train, y_train, observed_mask=observed_mask)
    dr_loss = dr_oracle._compute_expected_losses(  # noqa: SLF001
        candidate_feature_masks=candidate_mask,
        idx_nn=idx_nn,
        feature_count=2,
    )

    class_weight = torch.tensor(2.0)
    full_loss = class_weight * (-torch.log(torch.tensor(0.9)))
    overlap_loss = (
        class_weight * -torch.log(torch.tensor(0.9))
        - class_weight * torch.log(torch.tensor(0.6))
    ) / 2

    assert torch.allclose(mask_aware_loss, overlap_loss.unsqueeze(0))
    assert torch.allclose(dr_loss, full_loss.unsqueeze(0), atol=1e-6)
