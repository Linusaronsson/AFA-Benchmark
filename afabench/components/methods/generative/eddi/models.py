from typing import final, override

import torch
from torch import nn

from afabench.components.methods.rl.odin.models import PointNet
from afabench.core.types import FeatureMask, MaskedFeatures

type PartialVAEEncoding = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]
type PartialVAEForward = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


@final
class PartialVAE(nn.Module):
    """
    A partial VAE for masked data, as described in "EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE".

    To make the model work with different shapes of data, change the pointnet.
    """

    def __init__(
        self,
        pointnet: PointNet,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        """
        Initialize a partial VAE.

        Args:
        pointnet: maps unordered sets of features to a single vector
        encoder: a network that maps the output from the pointnet to input for mu_net and logvar_net
        decoder: the network to use for the decoder.

        """
        super().__init__()

        self.pointnet: PointNet = pointnet
        self.encoder: nn.Module = encoder
        self.decoder: nn.Module = decoder

    def encode(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> PartialVAEEncoding:
        pointnet_output = self.pointnet(masked_features, feature_mask)
        encoding = self.encoder(pointnet_output)

        mu = encoding[:, : encoding.shape[1] // 2]
        logvar = encoding[:, encoding.shape[1] // 2 :]
        # logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)

        return encoding, mu, logvar, z

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> PartialVAEForward:
        # Encode the masked features
        encoding, mu, logvar, z = self.encode(masked_features, feature_mask)

        # Decode
        x_hat = self.decoder(z)

        return encoding, mu, logvar, z, x_hat

    def impute(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> torch.Tensor:
        """Impute using a partial input."""
        _, _, _, _, recon = self.forward(masked_features, feature_mask)
        return recon
