from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from afabench.afa_oracle.aaco_core import AACOOracle
from afabench.common.initializers.cube_nm_ar_initializer import (
    CubeNMARInitializer,
)
from afabench.common.initializers.random_initializer import RandomInitializer
from scripts.train.aaco import (
    _apply_pvae_action_restoration,
    _apply_train_missingness,
    _derive_train_support_masks,
    _prepare_train_matrix,
)

if TYPE_CHECKING:
    from afabench.afa_rl.zannone2019.models import (
        Zannone2019PretrainingModel,
    )
    from afabench.common.config_classes import InitializerConfig


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


class _ToyPVAE:
    n_classes: int = 2

    def __init__(self) -> None:
        self.seen_masked_features: torch.Tensor | None = None
        self.seen_feature_mask: torch.Tensor | None = None
        self.seen_n_classes: int | None = None
        self.seen_label: torch.Tensor | None = None

    def masked_reconstruction(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,
        n_classes: int,
        label: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.seen_masked_features = masked_features.clone()
        self.seen_feature_mask = feature_mask.clone()
        self.seen_n_classes = n_classes
        self.seen_label = label
        reconstructed = torch.tensor(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
            dtype=masked_features.dtype,
            device=masked_features.device,
        )
        z = torch.zeros(
            (masked_features.shape[0], 1), device=masked_features.device
        )
        return z, reconstructed


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


def test_aaco_pvae_restoration_imputes_blocked_features_and_restores_support() -> (
    None
):
    x_train = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    train_support_mask = cast(
        "torch.BoolTensor",
        torch.tensor(
            [[True, False, True], [False, True, False]],
            dtype=torch.bool,
        ),
    )
    pvae = _ToyPVAE()

    restored, observed_mask = _apply_pvae_action_restoration(
        x_train=x_train,
        train_support_mask=train_support_mask,
        pvae_model=cast(
            "Zannone2019PretrainingModel",
            cast("object", pvae),
        ),
        hide_val=0.0,
    )

    expected = torch.tensor(
        [[1.0, 20.0, 3.0], [40.0, 5.0, 60.0]],
        dtype=torch.float32,
    )
    assert torch.equal(restored, expected)
    assert observed_mask is None
    assert pvae.seen_label is None
    assert pvae.seen_n_classes == pvae.n_classes
    assert pvae.seen_feature_mask is not None
    assert torch.equal(pvae.seen_feature_mask, train_support_mask)
    assert pvae.seen_masked_features is not None
    assert torch.equal(
        pvae.seen_masked_features,
        torch.tensor(
            [[1.0, 0.0, 3.0], [0.0, 5.0, 0.0]],
            dtype=torch.float32,
        ),
    )


def test_prepare_train_matrix_accepts_hydra_string_pvae_path(
    monkeypatch,  # noqa: ANN001
) -> None:
    x_train = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=torch.float32,
    )
    support = torch.tensor(
        [[True, False, True], [False, True, False]],
        dtype=torch.bool,
    )

    monkeypatch.setattr(
        "scripts.train.aaco.get_afa_initializer_from_config",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "scripts.train.aaco._derive_train_support_masks",
        lambda _initializer, **_kwargs: (support, support),
    )

    seen_path: Path | None = None

    def _fake_load_pvae(path: Path, *, device: torch.device) -> _ToyPVAE:
        nonlocal seen_path
        del device
        seen_path = path
        return _ToyPVAE()

    monkeypatch.setattr("scripts.train.aaco._load_pvae_model", _fake_load_pvae)

    restored, observed_mask, _initial_fraction, _support_fraction = (
        _prepare_train_matrix(
            x_train=x_train,
            feature_shape=torch.Size((3,)),
            initializer_cfg=cast("InitializerConfig", cast("object", None)),
            seed=0,
            mode="pvae_restore_actions",
            hide_val=0.0,
            device=torch.device("cpu"),
            pretrained_model_bundle_path="pvae.bundle",
        )
    )

    assert seen_path == Path("pvae.bundle")
    assert observed_mask is None
    assert torch.equal(
        restored,
        torch.tensor([[1.0, 20.0, 3.0], [40.0, 5.0, 60.0]]),
    )


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
