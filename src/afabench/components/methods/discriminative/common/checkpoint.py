"""
The architecture record of a GDFS or DIME checkpoint.

Both methods save a policy network and a classifier to `model.pt` together
with an `architecture` dict that says how to rebuild them. `NetworkPairSpec`
is that dict as a typed value: `to_dict` writes it, `from_dict` reads it
(defaulting a missing `n_selections` to `d_in`), and `build_networks`
rebuilds the untrained pair that the saved state dicts fit.

The dict names the policy network's hidden layers after the method's own
name for it (`selector` for GDFS, `value_network` for DIME), so the two
checkpoint formats stay distinct. The built-in classifier's fields use
`classifier` throughout; bundles written with the old spelling are not
supported.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Self

import torch
from torch import nn
from torchrl.modules import MLP

from afabench.components.methods.discriminative.common.models import (
    Classifier,
    ConvNet,
    ResNet18Backbone,
    resnet18,
    resnet50,
)

type Modality = Literal["tabular", "image"]

IMAGE_BACKBONES = ("resnet18", "resnet50")
TABULAR_FIELDS = (
    "d_in",
    "n_selections",
    "policy_hidden_layers",
    "classifier_hidden_layers",
    "dropout",
)
IMAGE_FIELDS = ("backbone", "image_size", "patch_size", "mask_width")


@dataclass(frozen=True)
class NetworkPairSpec:
    """
    How to rebuild a method's policy network and classifier.

    A tabular pair is two MLPs over the features with the feature mask
    appended; an image pair is a classifier and a ConvNet on ResNet
    backbones, selecting among `mask_width` squared patches. DIME shares
    the backbone; GDFS keeps the selector's backbone independent.
    """

    modality: Modality
    d_out: int
    d_in: int | None = None
    n_selections: int | None = None
    policy_hidden_layers: list[int] | None = None
    classifier_hidden_layers: list[int] | None = None
    dropout: float | None = None
    backbone: str | None = None
    image_size: int | None = None
    patch_size: int | None = None
    mask_width: int | None = None

    def __post_init__(self) -> None:
        required = (
            TABULAR_FIELDS if self.modality == "tabular" else IMAGE_FIELDS
        )
        for name in required:
            if getattr(self, name) is None:
                msg = f"a {self.modality} NetworkPairSpec needs {name}"
                raise ValueError(msg)
        if self.modality == "image" and self.backbone not in IMAGE_BACKBONES:
            msg = (
                f"unsupported backbone {self.backbone!r}; expected one of "
                f"{IMAGE_BACKBONES}"
            )
            raise ValueError(msg)

    def to_dict(self, policy_network_name: str) -> dict[str, Any]:
        """Write the checkpoint's `architecture` dict, keying the policy network's hidden layers by its name."""
        if self.modality == "tabular":
            return {
                "type": "mlp",
                "d_in": self.d_in,
                "d_out": self.d_out,
                "n_selections": self.n_selections,
                f"{policy_network_name}_hidden_layers": self.policy_hidden_layers,
                "classifier_hidden_layers": self.classifier_hidden_layers,
                "dropout": self.dropout,
                "model_type": "tabular",
            }
        return {
            "type": self.backbone,
            "backbone": self.backbone,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "mask_width": self.mask_width,
            "d_out": self.d_out,
            "model_type": "image",
        }

    @classmethod
    def from_dict(cls, arch: dict[str, Any], policy_network_name: str) -> Self:
        """Read a checkpoint's `architecture` dict written by `to_dict` or by the methods before it."""
        if arch["type"] == "mlp":
            d_in = int(arch["d_in"])
            n_selections = arch.get("n_selections")
            return cls(
                modality="tabular",
                d_in=d_in,
                d_out=int(arch["d_out"]),
                # Checkpoints from before grouped selections selected
                # among the features themselves
                n_selections=d_in
                if n_selections is None
                else int(n_selections),
                policy_hidden_layers=list(
                    arch[f"{policy_network_name}_hidden_layers"]
                ),
                classifier_hidden_layers=list(
                    arch["classifier_hidden_layers"]
                ),
                dropout=float(arch["dropout"]),
            )
        if arch["type"] in IMAGE_BACKBONES:
            return cls(
                modality="image",
                d_out=int(arch["d_out"]),
                backbone=str(arch["type"]),
                image_size=int(arch["image_size"]),
                patch_size=int(arch["patch_size"]),
                mask_width=int(arch["mask_width"]),
            )
        msg = (
            f"unrecognized checkpoint architecture type {arch['type']!r}; "
            f"expected 'mlp' or one of {IMAGE_BACKBONES}"
        )
        raise ValueError(msg)

    def build_networks(
        self, device: torch.device, *, share_image_backbone: bool = True
    ) -> tuple[nn.Module, nn.Module]:
        """Build the pair on `device`; GDFS needs an independent image backbone."""
        if self.modality == "tabular":
            assert self.d_in is not None
            assert self.n_selections is not None
            policy_network = MLP(
                in_features=self.d_in * 2,
                out_features=self.n_selections,
                num_cells=self.policy_hidden_layers,
                activation_class=nn.ReLU,
                dropout=self.dropout,
            )
            classifier = MLP(
                in_features=self.d_in * 2,
                out_features=self.d_out,
                num_cells=self.classifier_hidden_layers,
                activation_class=nn.ReLU,
                dropout=self.dropout,
            )
        else:
            base = (
                resnet18(pretrained=False)
                if self.backbone == "resnet18"
                else resnet50(pretrained=False)
            )
            backbone_net, expansion = ResNet18Backbone(base)
            classifier = Classifier(backbone_net, expansion, self.d_out)
            # GDFS trains two backbones. Reusing the module would let
            # the classifier's state dict overwrite the selector's (#117).
            policy_backbone = (
                backbone_net
                if share_image_backbone
                else deepcopy(backbone_net)
            )
            policy_network = ConvNet(policy_backbone, expansion, 0.5)
        return policy_network.to(device), classifier.to(device)
