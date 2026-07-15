import torch
from torch import nn

from afabench.components.methods.generative.eddi.afa_methods import (
    EDDIAFAMethod,
)


class FakeSampler(nn.Module):
    def forward(
        self, masked_features: torch.Tensor, feature_mask: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del feature_mask
        batch_size = masked_features.shape[0]
        z = torch.zeros(batch_size, 3, device=masked_features.device)
        return z, z, z, z, masked_features


class FakePredictor(nn.Module):
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.stack([z[:, 0], z[:, 0] + 1], dim=-1)


def test_eddi_predict_flattens_image_features() -> None:
    method = EDDIAFAMethod(
        sampler=FakeSampler(),
        predictor=FakePredictor(),
        num_classes=2,
        num_mc_samples=2,
    )
    masked_features = torch.zeros((2, 1, 28, 28))
    feature_mask = torch.zeros_like(masked_features, dtype=torch.bool)

    prediction = method.predict(
        masked_features=masked_features,
        feature_mask=feature_mask,
        feature_shape=torch.Size([1, 28, 28]),
    )

    assert prediction.shape == (2, 2)


def test_eddi_act_supports_synthetic_mnist_patch_selection() -> None:
    method = EDDIAFAMethod(
        sampler=FakeSampler(),
        predictor=FakePredictor(),
        num_classes=2,
        selection_costs=torch.ones(49),
        num_mc_samples=2,
    )
    masked_features = torch.zeros((2, 1, 28, 28))
    feature_mask = torch.zeros_like(masked_features, dtype=torch.bool)
    selection_mask = torch.zeros((2, 49), dtype=torch.bool)

    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        feature_shape=torch.Size([1, 28, 28]),
    )

    assert action.shape == (2, 1)
    assert torch.all((action >= 1) & (action <= 49))
