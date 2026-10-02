from copy import deepcopy

import pytest
import torch
from torch import nn

from afabench.components.methods.rl.odin.models import (
    ODINPretrainingModel,
    PartialVAE,
    PointNet,
    PointNetType,
)


def _legacy_forward(
    model: PointNet,
    features: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    identity = model.embedding_net(
        torch.arange(model.n_features).repeat(len(features), 1)
    )
    if model.pointnet_type == PointNetType.POINTNETPLUS:
        encoded = features.unsqueeze(-1) * identity
    else:
        encoded = torch.cat([features.unsqueeze(-1), identity], dim=-1)
    return (model.feature_map_encoder(encoded) * mask.unsqueeze(-1)).sum(1)


@pytest.mark.parametrize(
    ("pointnet_type", "input_size"),
    [(PointNetType.POINTNET, 4), (PointNetType.POINTNETPLUS, 3)],
)
def test_unique_identity_lookup_preserves_pointnet_update(
    pointnet_type: PointNetType,
    input_size: int,
) -> None:
    torch.manual_seed(0)
    template = PointNet(
        identity_size=3,
        n_features=5,
        feature_map_encoder=nn.Sequential(nn.Linear(input_size, 4), nn.ReLU()),
        pointnet_type=pointnet_type,
        max_embedding_norm=1.0,
    )
    features = torch.randn(7, 5)
    mask = torch.rand(7, 5) > 0.25
    legacy = deepcopy(template)
    unique = deepcopy(template)

    legacy_output = _legacy_forward(legacy, features, mask)
    unique_output = unique(features, mask)
    legacy_output.sum().backward()
    unique_output.sum().backward()

    torch.testing.assert_close(unique_output, legacy_output, rtol=0, atol=0)
    for legacy_parameter, unique_parameter in zip(
        legacy.parameters(), unique.parameters(), strict=True
    ):
        torch.testing.assert_close(
            unique_parameter,
            legacy_parameter,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            unique_parameter.grad,
            legacy_parameter.grad,
        )


def test_partial_vae_clamps_sampling_but_not_kl_logvar() -> None:
    model = PartialVAE(
        pointnet=PointNet(
            identity_size=1,
            n_features=1,
            feature_map_encoder=nn.Identity(),
            pointnet_type=PointNetType.POINTNETPLUS,
        ),
        encoder=nn.Linear(1, 2),
        decoder=nn.Identity(),
        latent_size=1,
    )
    with torch.no_grad():
        model.encoder.weight.zero_()
        model.encoder.bias.copy_(torch.tensor([0.0, 1000.0]))

    torch.manual_seed(1)
    _encoding, mu, logvar, sampled = model.encode(
        torch.ones((2, 1)), torch.ones((2, 1), dtype=torch.bool)
    )
    # Same seed, same single draw: recover the noise the sampler used.
    torch.manual_seed(1)
    eps = torch.randn_like(sampled)

    # Without clamping the sampling std, exp(0.5 * 1000) overflows to inf.
    # The clamp caps logvar at 20, so the std is exactly exp(10).
    assert torch.isfinite(sampled).all()
    torch.testing.assert_close(
        sampled, mu + torch.exp(torch.tensor(10.0)) * eps
    )
    # The KL term is computed downstream from this returned logvar, so it
    # must stay unclamped even though sampling clamps it internally.
    torch.testing.assert_close(logvar, torch.full_like(logvar, 1000.0))

    # Feed the returned logvar through the pretraining loss: exp(1000)
    # overflows, so the KL term is infinite. Had the clamp leaked into the
    # returned value, the KL would be finite.
    pretraining_model = ODINPretrainingModel(
        partial_vae=model,
        classifier=nn.Identity(),
        class_probabilities=torch.tensor([0.5, 0.5]),
        min_masking_probability=0.0,
        max_masking_probability=1.0,
        lr=1e-3,
        start_kl_scaling_factor=1.0,
        end_kl_scaling_factor=1.0,
        n_annealing_epochs=1,
        classifier_loss_scaling_factor=1.0,
    )
    _total, _recon, kl = pretraining_model.partial_vae_loss_function(
        torch.zeros((2, 1)), torch.zeros((2, 1)), mu, logvar
    )
    assert torch.isinf(kl)
    _total, _recon, kl_clamped = pretraining_model.partial_vae_loss_function(
        torch.zeros((2, 1)), torch.zeros((2, 1)), mu, logvar.clamp_max(20.0)
    )
    assert torch.isfinite(kl_clamped)
