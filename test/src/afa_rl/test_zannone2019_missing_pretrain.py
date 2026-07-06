from __future__ import annotations

import torch
from torch import nn
from torchrl.modules import MLP

from afabench.afa_rl.zannone2019.models import (
    PartialVAE,
    PointNet,
    PointNetType,
    Zannone2019PretrainingModel,
)

_N_FEATURES = 4
_N_CLASSES = 2
_LATENT_SIZE = 3


def _make_model() -> Zannone2019PretrainingModel:
    pointnet_output_size = 8
    identity_size = 4
    pointnet = PointNet(
        identity_size=identity_size,
        n_features=_N_FEATURES + _N_CLASSES,
        max_embedding_norm=1.0,
        feature_map_encoder=MLP(
            in_features=identity_size + 1,
            out_features=pointnet_output_size,
            num_cells=[8],
            dropout=0.0,
            activation_class=nn.ReLU,
        ),
        pointnet_type=PointNetType.POINTNET,
    )
    partial_vae = PartialVAE(
        pointnet=pointnet,
        encoder=MLP(
            in_features=pointnet_output_size,
            out_features=2 * _LATENT_SIZE,
            num_cells=[8],
            dropout=0.0,
            activation_class=nn.ReLU,
        ),
        decoder=MLP(
            in_features=_LATENT_SIZE,
            out_features=_N_FEATURES,
            num_cells=[8],
            dropout=0.0,
            activation_class=nn.ReLU,
        ),
        latent_size=_LATENT_SIZE,
    )
    model = Zannone2019PretrainingModel(
        partial_vae=partial_vae,
        classifier=MLP(
            in_features=_LATENT_SIZE,
            out_features=_N_CLASSES,
            num_cells=[8],
            dropout=0.0,
            activation_class=nn.ReLU,
        ),
        class_probabilities=torch.full((_N_CLASSES,), 0.5),
        min_masking_probability=0.2,
        max_masking_probability=0.8,
        lr=1e-3,
        start_kl_scaling_factor=0.1,
        end_kl_scaling_factor=0.1,
        n_annealing_epochs=1,
        classifier_loss_scaling_factor=1.0,
    )
    # No Trainer is attached in these tests; capture logs instead.
    model.log = lambda *_args, **_kwargs: None  # pyright: ignore[reportAttributeAccessIssue]
    return model


def _make_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.tensor(
        [
            [1.0, -2.0, 0.5, 3.0],
            [0.0, 1.5, -1.0, 2.0],
            [2.0, 0.5, 1.0, -3.0],
        ],
        dtype=torch.float32,
    )
    label = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=torch.float32,
    )
    mechanism_mask = torch.tensor(
        [
            [True, False, True, True],
            [True, True, False, False],
            [False, True, True, True],
        ],
        dtype=torch.bool,
    )
    return features, label, mechanism_mask


def _training_loss(
    model: Zannone2019PretrainingModel,
    batch: tuple[torch.Tensor, ...],
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    with torch.no_grad():
        return model.training_step(batch, 0)


def test_training_loss_invariant_to_mechanism_missing_values() -> None:
    """The true values of mechanism-missing entries must not leak into pretraining: neither through the input nor through the reconstruction loss."""
    torch.manual_seed(0)
    model = _make_model()
    features, label, mechanism_mask = _make_batch()

    perturbed = features.clone()
    perturbed[~mechanism_mask] += 100.0

    for seed in range(3):
        loss_original = _training_loss(
            model, (features, label, mechanism_mask), seed
        )
        loss_perturbed = _training_loss(
            model, (perturbed, label, mechanism_mask), seed
        )
        assert torch.allclose(loss_original, loss_perturbed), (
            "Training loss changed when only mechanism-missing entries "
            "were perturbed - hidden values are leaking."
        )


def test_training_loss_sensitive_to_available_values() -> None:
    """Sanity counterpart: perturbing entries that exist in the record must change the loss."""
    torch.manual_seed(0)
    model = _make_model()
    features, label, mechanism_mask = _make_batch()

    perturbed = features.clone()
    perturbed[mechanism_mask] += 100.0

    loss_original = _training_loss(model, (features, label, mechanism_mask), 0)
    loss_perturbed = _training_loss(
        model, (perturbed, label, mechanism_mask), 0
    )
    assert not torch.allclose(loss_original, loss_perturbed)


def test_two_tuple_batch_matches_all_available_mechanism_mask() -> None:
    """Batches without a mechanism mask must behave exactly like an all-available mask (backward compatibility)."""
    torch.manual_seed(0)
    model = _make_model()
    features, label, _mechanism_mask = _make_batch()
    all_available = torch.ones_like(features, dtype=torch.bool)

    for seed in range(3):
        loss_two_tuple = _training_loss(model, (features, label), seed)
        loss_all_available = _training_loss(
            model, (features, label, all_available), seed
        )
        assert torch.allclose(loss_two_tuple, loss_all_available)


def test_validation_metrics_invariant_to_mechanism_missing_values() -> None:
    torch.manual_seed(0)
    model = _make_model()
    features, label, mechanism_mask = _make_batch()

    perturbed = features.clone()
    perturbed[~mechanism_mask] += 100.0

    def _validation_metrics(
        batch: tuple[torch.Tensor, ...], seed: int
    ) -> dict[str, float]:
        logged: dict[str, float] = {}

        def _record(name, value, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            del args, kwargs
            logged[name] = float(value)

        model.log = _record  # pyright: ignore[reportAttributeAccessIssue]
        torch.manual_seed(seed)
        with torch.no_grad():
            model.validation_step(batch, 0)
        return logged

    metrics_original = _validation_metrics(
        (features, label, mechanism_mask), 0
    )
    metrics_perturbed = _validation_metrics(
        (perturbed, label, mechanism_mask), 0
    )
    assert metrics_original.keys() == metrics_perturbed.keys()
    for name, value in metrics_original.items():
        assert abs(value - metrics_perturbed[name]) < 1e-5, (
            f"Validation metric {name} changed when only mechanism-missing "
            "entries were perturbed."
        )
