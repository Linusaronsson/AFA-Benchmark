from copy import deepcopy

import pytest
import torch
from torch import nn

from afabench.components.methods.rl.odin.models import (
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


def test_partial_vae_bounds_sampling_variance() -> None:
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
        model.encoder.bias.copy_(torch.tensor([0.0, 100.0]))

    expected_generator = torch.Generator().manual_seed(1)
    expected = torch.randn((2, 1), generator=expected_generator) * torch.exp(
        torch.tensor(10.0)
    )
    generator = torch.Generator().manual_seed(1)
    _, _, logvar, sampled = model.encode(
        torch.ones((2, 1)), torch.ones((2, 1), dtype=torch.bool), generator
    )

    torch.testing.assert_close(logvar, torch.full_like(logvar, 100.0))
    torch.testing.assert_close(sampled, expected)
