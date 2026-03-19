import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Self, Any, Callable

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchrl.modules import MLP

from afabench.afa_discriminative.utils import (
    restore_parameters,
    to_class_indices,
)
from afabench.afa_rl.common.utils import mask_data
from afabench.common.custom_types import AFAInitializer

models_dir = "./models/pretrained_resnet_models"
model_name = {
    "resnet18": "resnet18-5c106cde.pth",
    "resnet50": "resnet50-19c8e357.pth",
}

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def ResNet18Backbone(model):
    # Remove avgpool and fc
    backbone_modules = nn.ModuleList(model.children())[:-2]
    return nn.Sequential(*backbone_modules), model.expansion


def resnet18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        try:
            ckpt_path = os.path.join(models_dir, model_name["resnet18"])
            state_dict = torch.load(ckpt_path, map_location="cpu")
        except FileNotFoundError:
            from torch.hub import load_state_dict_from_url

            state_dict = load_state_dict_from_url(
                model_urls["resnet18"],
                weights_only=False,
            )
        model.load_state_dict(state_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            ckpt_path = os.path.join(models_dir, model_name["resnet50"])
            state_dict = torch.load(ckpt_path, map_location="cpu")
        except FileNotFoundError:
            from torch.hub import load_state_dict_from_url
            state_dict = load_state_dict_from_url(
                model_urls["resnet50"],
                weights_only=False,
            )
        model.load_state_dict(state_dict)
    return model


def make_layer(block, in_planes, planes, num_blocks, stride, expansion):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
        if stride >= 1:
            downsample = None

            if stride != 1 or in_planes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers.append(block(in_planes, planes, stride, downsample))
        elif expansion == 1:
            layers.append(UpsamplingBlock(in_planes, planes))
        else:
            layers.append(UpsamplingBottleneckBlock(in_planes, planes))
        in_planes = planes * block.expansion
    return nn.Sequential(*layers)


class UpsamplingBlock(nn.Module):
    """Custom residual block for performing upsampling."""

    expansion = 1

    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                self.expansion * planes,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(self.expansion * planes),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class UpsamplingBottleneckBlock(nn.Module):
    """Custom residual block for performing upsampling."""

    expansion = 4

    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_planes, planes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes,
                planes * self.expansion,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(planes * self.expansion),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class MaskingPretrainer(nn.Module):
    """Pretrain model with missing features."""

    def __init__(self,
        model,
        mask_layer,
        initializer: AFAInitializer | None = None,
        feature_shape: torch.Size | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.mask_layer = mask_layer
        self.initializer = initializer
        self.feature_shape = feature_shape

    def fit(
        self,
        train_loader,
        val_loader,
        lr,
        nepochs,
        loss_fn,
        val_loss_fn=None,
        val_loss_mode=None,
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        early_stopping_epochs=None,
        verbose=True,
        min_mask=0.1,
        max_mask=0.9,
    ):
        """Train model."""
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = "min"
        elif val_loss_mode is None:
            raise ValueError(
                "must specify val_loss_mode (min or max) when validation_loss_fn is specified"
            )

        # Set up optimizer and lr scheduler.
        model = self.model
        mask_layer = self.mask_layer
        device = next(model.parameters()).device
        opt = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=val_loss_mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        # For tracking best model and early stopping.
        best_model: nn.Module | None = None
        num_bad_epochs = 0
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1

        for epoch in range(nepochs):
            # Switch model to training mode.
            model.train()
            epoch_train_loss = 0.0
            for batch in train_loader:
                if len(batch) == 3:
                    x, y, forbidden_feat = batch
                else:
                    x, y = batch
                    raise RuntimeError(
                        "Forbidden mask missing from pretraining dataloader"
                    )
                x = x.to(device)
                y = to_class_indices(y).to(device)
                forbidden_feat = forbidden_feat.bool().to(device)

                # allowed_feat = torch.ones_like(x, dtype=torch.bool, device=device)
                allowed_feat = ~forbidden_feat

                # Generate missingness.
                p = min_mask + torch.rand(1).item() * (max_mask - min_mask)
                if x.dim() == 4:
                    n = self.mask_layer.mask_size
                    m = (torch.rand(x.size(0), n, device=device) < p).float()
                else:
                    _, m, _ = mask_data(x, p)
                    m = m & allowed_feat
                    m = m.float()

                # Calculate loss.
                x_masked = mask_layer(x, m)
                pred = model(x_masked)
                loss = loss_fn(pred, y)

                # Take gradient step.
                opt.zero_grad()
                loss.backward()
                opt.step()
                # model.zero_grad()
                epoch_train_loss += loss.item()

            avg_train = epoch_train_loss / len(train_loader)

            # Calculate validation loss.
            model.eval()
            with torch.no_grad():
                # For mean loss.
                pred_list = []
                label_list = []

                for batch in val_loader:
                    if len(batch) == 3:
                        x, y, forbidden_feat = batch
                    else:
                        x, y = batch
                        raise RuntimeError(
                            "Forbidden mask missing from pretraining dataloader"
                        )
                    x = x.to(device)
                    y = to_class_indices(y).to(device)
                    forbidden_feat = forbidden_feat.to(device).bool()
                    allowed_feat = ~forbidden_feat

                    # Generate missingness.
                    p = min_mask + torch.rand(1).item() * (max_mask - min_mask)

                    # Calculate prediction.
                    if x.dim() == 4:
                        n = self.mask_layer.mask_size
                        m = (
                            torch.rand(x.size(0), n, device=device) < p
                        ).float()
                    else:
                        _, m, _ = mask_data(x, p)
                        m = m & allowed_feat
                        m = m.float()
                    x_masked = mask_layer(x, m)
                    pred = model(x_masked)
                    pred_list.append(pred)
                    label_list.append(y)

                # Calculate loss.
                y = torch.cat(label_list, 0)
                pred = torch.cat(pred_list, 0)
                val_loss = val_loss_fn(pred, y).item()

            # Print progress.
            if verbose:
                print(f"{'-' * 8}Epoch {epoch + 1}{'-' * 8}")
                print(f"Train loss = {avg_train:.4f}\n")
                print(f"Val loss = {val_loss:.4f}\n")

            # Update scheduler.
            scheduler.step(val_loss)

            # Check if best model.
            if val_loss == scheduler.best:
                best_model = deepcopy(model)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            # Early stopping.
            if num_bad_epochs > early_stopping_epochs:
                if verbose:
                    print(f"Stopping early at epoch {epoch + 1}")
                break

        # Copy parameters from best model.
        assert best_model is not None
        restore_parameters(model, best_model)

    def forward(self, x, mask):
        """
        Generate model prediction.

        Args:
          x:
          mask:

        """
        x_masked = self.mask_layer(x, mask)
        return self.model(x_masked)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.expansion = block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class Predictor(nn.Module):
    def __init__(self, backbone, expansion, num_classes=10):
        super(Predictor, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Conv2d(512 * expansion, 256, kernel_size=3)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(256, num_classes)
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(self.bn(self.conv(x)))
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, backbone, expansion=1, block_layer_stride=0.5):
        super(ConvNet, self).__init__()
        self.backbone = backbone
        if expansion == 1:
            # BasicBlock
            self.block_layer = make_layer(
                BasicBlock,
                512,
                256,
                2,
                stride=block_layer_stride,
                expansion=expansion,
            )
            self.conv = nn.Conv2d(256, 1, 1)
        else:
            # Bottleneck block
            # BasicBlock
            self.block_layer = make_layer(
                Bottleneck,
                2048,
                512,
                2,
                stride=block_layer_stride,
                expansion=expansion,
            )
            self.conv = nn.Conv2d(2048, 1, 1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.backbone(x)
        x = self.block_layer(x)
        # print(x.shape)
        x = self.conv(x)
        x = self.flatten(x)
        return x


def build_mlp(arch: dict[str, Any]) -> nn.Module:
    return MLP(
        in_features=arch["in_features"],
        out_features=arch["out_features"],
        num_cells=arch["hidden_units"],
        activation_class=getattr(nn, arch["activation"]),
        dropout=arch["dropout"],
    )

def build_resnet18_predictor(arch: dict[str, Any]) -> nn.Module:
    base = resnet18(pretrained=bool(arch.get("pretrained", True)))
    backbone, expansion = ResNet18Backbone(base)
    return Predictor(backbone, expansion, num_classes=int(arch["d_out"]))

def build_resnet50_predictor(arch: dict[str, Any]) -> nn.Module:
    base = resnet50(pretrained=bool(arch.get("pretrained", True)))
    backbone, expansion = ResNet18Backbone(base)
    return Predictor(backbone, expansion, num_classes=int(arch["d_out"]))

_BUILDERS: dict[str, Callable[[dict[str, Any]], nn.Module]] = {
    "mlp": build_mlp,
    "resnet18": build_resnet18_predictor,
    "resnet50": build_resnet50_predictor,
}


class GreedyAFAClassifier:
    def __init__(
        self,
        predictor: nn.Module,
        architecture: dict[str, Any],
        device: torch.device,
    ):
        self.predictor: nn.Module = predictor.to(device)
        self.predictor.eval()
        self.architecture: dict[str, Any] = architecture
        self._device: torch.device = device

    def save(self, path: Path) -> None:
        """Save all necessary files into the given path."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "predictor_state_dict": self.predictor.state_dict(),
                "architecture": self.architecture,
            },
            path / "model.pt",
        )

    @classmethod
    def load(
        cls,
        path: Path,
        map_location: torch.device,
    ) -> Self:
        checkpoint = torch.load(path / "model.pt", map_location=map_location, weights_only=False)
        arch: dict[str, Any] = checkpoint["architecture"]
        state_dict = checkpoint["predictor_state_dict"]
        model_type = arch.get("type")
        if not model_type:
            raise ValueError("Checkpoint missing architecture['type']")

        builder = _BUILDERS.get(str(model_type))
        if builder is None:
            raise ValueError(f"Unsupported model type: {model_type}")

        predictor = builder(arch)
        predictor.load_state_dict(state_dict)
        predictor.eval()

        return cls(predictor=predictor, architecture=arch, device=map_location)


class NotMIWAE(nn.Module):
    """
    - encoder: x -> q(z|x)
    - decoder: z -> mu
    - p(x|z) = Normal(mu, exp(logstd)) with global scalar logstd
    - missing process: self-masking logistic model
    """
    def __init__(
        self,
        d_in: int,
        n_latent: int,
        n_hidden: int = 128,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_latent = n_latent

        act = activation

        self.encoder = nn.Sequential(
            nn.Linear(d_in, n_hidden),
            act(),
            nn.Linear(n_hidden, n_hidden),
            act(),
        )
        self.q_mu = nn.Linear(n_hidden, n_latent)
        self.q_logstd = nn.Linear(n_hidden, n_latent)

        # z -> mu
        self.mu = nn.Linear(n_latent, d_in)
        # Global scalar log std for p(x|z)
        self.logstd = nn.Parameter(torch.zeros(()))

        # Self-masking missingness model parameters
        self.raw_W = nn.Parameter(torch.zeros(1, 1, d_in))
        self.b = nn.Parameter(torch.zeros(1, 1, d_in))

    def forward(
        self,
        x_filled: torch.Tensor,
        s: torch.Tensor,
        n_samples: int,
    ) -> dict[str, torch.Tensor]:
        """
        x_filled: [B, D], zero-filled at missing positions
        s: [B, D], 1 for observed, 0 for missing
        """
        h = self.encoder(x_filled)
        q_mu = self.q_mu(h)
        q_logstd = torch.clamp(self.q_logstd(h), min=-10.0, max=10.0)
        q_std = torch.exp(q_logstd)

        # Reparameterized samples: [B, L, K]
        eps = torch.randn(
            x_filled.size(0), n_samples, self.n_latent, device=x_filled.device
        )
        z = q_mu[:, None, :] + eps * q_std[:, None, :]

        # Decoder mean: [B, L, D]
        mu = self.mu(z)
        px_std = torch.exp(self.logstd)

        x_exp = x_filled[:, None, :]
        s_exp = s[:, None, :]

        # log p(x_o | z)
        p_x_given_z = torch.distributions.Normal(loc=mu, scale=px_std)
        log_p_x_given_z = (p_x_given_z.log_prob(x_exp) * s_exp).sum(dim=-1)

        # log q(z | x)
        q_z = torch.distributions.Normal(
            loc=q_mu[:, None, :],
            scale=q_std[:, None, :],
        )
        log_q_z_given_x = q_z.log_prob(z).sum(dim=-1)

        # log p(z)
        prior = torch.distributions.Normal(
            loc=torch.zeros_like(z),
            scale=torch.ones_like(z),
        )
        log_p_z = prior.log_prob(z).sum(dim=-1)

        # Missingness model p(s | x)
        # notebook: W = -softplus(W), logits = W * (x_mixed - b)
        W = -F.softplus(self.raw_W)
        # [B, L, D], observed -> x, missing -> mu
        l_out_mixed = mu * (1.0 - s_exp) + x_exp * s_exp
        logits = W * (l_out_mixed - self.b)

        log_p_s_given_x = -F.binary_cross_entropy_with_logits(
            logits,
            s_exp.expand_as(logits),
            reduction="none",
        ).sum(dim=-1)

        # not-MIWAE bound
        log_w = log_p_x_given_z + log_p_s_given_x + log_p_z - log_q_z_given_x
        log_avg_weight = torch.logsumexp(log_w, dim=1) - math.log(n_samples)
        notmiwae = log_avg_weight.mean()
        loss = -notmiwae

        return {
            "loss": loss,
            "notmiwae": notmiwae,
            "mu": mu,
            "log_p_x_given_z": log_p_x_given_z,
            "log_p_s_given_x": log_p_s_given_x,
            "log_p_z": log_p_z,
            "log_q_z_given_x": log_q_z_given_x,
        }

    @torch.no_grad()
    def impute(
        self,
        x_filled: torch.Tensor,
        s: torch.Tensor,
        n_samples: int = 1000,
    ) -> torch.Tensor:
        """
        Returns posterior-weighted imputations xm, shape [B, D].
        """
        out = self.forward(x_filled=x_filled, s=s, n_samples=n_samples)
        mu = out["mu"]  # [B, L, D]

        log_w = (
            out["log_p_x_given_z"]
            + out["log_p_s_given_x"]
            + out["log_p_z"]
            - out["log_q_z_given_x"]
        )
        w = torch.softmax(log_w, dim=1)  # [B, L]
        xm = (mu * w.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return xm

    @torch.no_grad()
    def feature_observation_probs(
        self,
        x_filled: torch.Tensor,
        s: torch.Tensor,
        n_samples: int = 100,
    ) -> torch.Tensor:
        h = self.encoder(x_filled)
        q_mu = self.q_mu(h)
        q_logstd = torch.clamp(self.q_logstd(h), min=-10.0, max=10.0)
        q_std = torch.exp(q_logstd)

        eps = torch.randn(
            x_filled.size(0), n_samples, self.n_latent, device=x_filled.device
        )
        z = q_mu[:, None, :] + eps * q_std[:, None, :]

        # [B, L, D]
        mu = self.mu(z)
        # [B, 1, D]
        x_exp = x_filled[:, None, :]
        s_exp = s[:, None, :]

        W = -F.softplus(self.raw_W)
        l_out_mixed = mu * (1.0 - s_exp) + x_exp * s_exp
        logits = W * (l_out_mixed - self.b)

        # [B, L, D]
        probs = torch.sigmoid(logits)
        return probs.mean(dim=1)


def train_notmiwae(
    model: NotMIWAE,
    x_train: torch.Tensor,
    s_train: torch.Tensor,
    x_val: torch.Tensor | None = None,
    s_val: torch.Tensor | None = None,
    *,
    lr: float = 1e-3,
    batch_size: int = 128,
    max_iter: int = 30000,
    n_samples: int = 20,
    eval_every: int = 100,
) -> NotMIWAE:
    device = next(model.parameters()).device
    model.train()

    train_ds = TensorDataset(x_train, s_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")

    step = 0
    while step < max_iter:
        for xb, sb in train_loader:
            xb = xb.to(device)
            sb = sb.to(device)

            out = model(x_filled=xb, s=sb, n_samples=n_samples)
            loss = out["loss"]

            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % eval_every == 0:
                model.eval()
                with torch.no_grad():
                    if x_val is None or s_val is None:
                        val_out = model(
                            x_filled=x_train.to(device),
                            s=s_train.to(device),
                            n_samples=n_samples,
                        )
                    else:
                        val_out = model(
                            x_filled=x_val.to(device),
                            s=s_val.to(device),
                            n_samples=n_samples,
                        )
                    val_loss = float(val_out["loss"].item())

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = deepcopy(model.state_dict())

                model.train()

            step += 1
            if step >= max_iter:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model
