"""
The modality-specific setup behind the GDFS, DIME and CAE trainers.

A method trains the same way on tabular and image datasets except for how
the data is batched, how the policy network is built and probed, and how
features are masked for the networks. `prepare_modality` reads the
pretrained model's architecture and answers a `ModalitySetup` with those
parts settled: `TabularSetup` for an MLP pretrained model, `ImageSetup`
for a ResNet one. It is an optional library helper that each method's own
training script calls (ADR 0001). CAE, which trains no policy network on a
pretrained model, uses only `prepare_datasets`, `image_loaders` and
`load_backbone`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, override

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchrl.modules import MLP

from afabench.components.methods.discriminative.common.checkpoint import (
    IMAGE_BACKBONES,
    NetworkPairSpec,
)
from afabench.components.methods.discriminative.common.datasets import (
    prepare_datasets,
)
from afabench.components.methods.discriminative.common.models import (
    Classifier,
    ConvNet,
    ResNet18Backbone,
    resnet18,
    resnet50,
)
from afabench.components.methods.discriminative.common.utils import (
    MaskLayer,
    MaskLayer2d,
    tie_first_k_linears_by_module,
)
from afabench.core.types import AFADataset
from afabench.core.utils import get_class_frequencies
from afabench.fit.inputs import FitInputs
from afabench.fit.smoke_test import dataset_subset, training_batch_size

type ClassWeights = Float[Tensor, "n_classes"]  # noqa: F821
type FeatureCosts = Float[Tensor, "*feature_shape"]
type SelectionCosts = Float[Tensor, "n_selections"]  # noqa: F821


class MLPPolicyArchitecture(Protocol):
    """A method's tabular architecture config: its MLP policy network."""

    @property
    def activation(self) -> str: ...

    @property
    def hidden_units(self) -> list[int]: ...

    @property
    def dropout(self) -> float: ...


class ResNetPolicyArchitecture(Protocol):
    """A method's image architecture config: its ConvNet's ResNet backbone."""

    @property
    def backbone_type(self) -> str: ...


class ModalityConfig(Protocol):
    """The training-config fields that the setup reads."""

    @property
    def tabular_architecture(self) -> MLPPolicyArchitecture: ...

    @property
    def image_architecture(self) -> ResNetPolicyArchitecture: ...

    @property
    def batch_size(self) -> int: ...

    @property
    def smoke_test(self) -> bool: ...

    @property
    def device(self) -> str: ...


@dataclass(frozen=True, kw_only=True)
class ModalitySetup(ABC):
    """The data, dimensions and networks one modality trains with."""

    device: torch.device
    train_loader: DataLoader[Any]
    val_loader: DataLoader[Any]
    d_in: int
    d_out: int
    n_selections: int
    mask_layer: MaskLayer | MaskLayer2d
    class_weights: ClassWeights | None
    """The training loss's class weights, on the device; None weighs classes equally."""
    feature_costs: FeatureCosts | None
    """What training charges per feature; None charges unit costs."""
    selection_costs: SelectionCosts | None
    """What the trained method scales its scores by; None for unit costs."""

    @abstractmethod
    def policy_network(
        self, *, share_with: nn.Module | None = None
    ) -> nn.Module:
        """
        Build the policy network on the device, with one output per selection.

        With `share_with`, the network shares its feature extraction with
        that built-in classifier, as DIME's value network does.
        """

    @abstractmethod
    def check_policy_output(self, policy_network: nn.Module) -> None:
        """Raise ValueError unless `policy_network` scores every selection."""

    @abstractmethod
    def min_lr(
        self, configured: float | None, *, upstream_default: float
    ) -> float:
        """Resolve the scheduler's minimum learning rate from the config's."""

    @abstractmethod
    def network_pair_spec(self) -> NetworkPairSpec:
        """Describe the trained pair as the checkpoint's architecture record."""

    def _check_n_scores(self, n_scores: int) -> None:
        if n_scores != self.n_selections:
            msg = (
                f"The policy network outputs {n_scores} scores for "
                f"{self.n_selections} selections"
            )
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class TabularSetup(ModalitySetup):
    """Flat features with the feature mask appended, for MLP networks."""

    architecture: MLPPolicyArchitecture

    @override
    def policy_network(
        self, *, share_with: nn.Module | None = None
    ) -> nn.Module:
        policy_network = MLP(
            in_features=self.d_in * 2,
            out_features=self.n_selections,
            num_cells=self.architecture.hidden_units,
            activation_class=getattr(nn, self.architecture.activation),
            dropout=self.architecture.dropout,
        ).to(self.device)
        if share_with is not None:
            tie_first_k_linears_by_module(share_with, policy_network, k=2)
        return policy_network

    @override
    def check_policy_output(self, policy_network: nn.Module) -> None:
        # In eval mode, so that dropout draws nothing from the RNG that
        # training then uses
        was_training = policy_network.training
        policy_network.eval()
        features = torch.zeros(1, self.d_in, device=self.device)
        with torch.no_grad():
            scores = policy_network(
                self.mask_layer(features, torch.zeros_like(features))
            )
        policy_network.train(was_training)
        self._check_n_scores(scores.shape[1])

    @override
    def min_lr(
        self, configured: float | None, *, upstream_default: float
    ) -> float:
        # The tabular experiment configs set no min_lr
        return upstream_default if configured is None else configured

    @override
    def network_pair_spec(self) -> NetworkPairSpec:
        return NetworkPairSpec(
            modality="tabular",
            d_in=self.d_in,
            d_out=self.d_out,
            n_selections=self.n_selections,
            policy_hidden_layers=self.architecture.hidden_units,
            classifier_hidden_layers=self.architecture.hidden_units,
            dropout=self.architecture.dropout,
        )


@dataclass(frozen=True, kw_only=True)
class ImageSetup(ModalitySetup):
    """Images selected patch by patch, for ConvNet networks on a ResNet."""

    architecture: ResNetPolicyArchitecture
    image_size: int
    patch_size: int
    mask_width: int

    @override
    def policy_network(
        self, *, share_with: nn.Module | None = None
    ) -> nn.Module:
        backbone, expansion = load_backbone(self.architecture.backbone_type)
        if share_with is not None:
            if not isinstance(share_with, Classifier):
                msg = (
                    "An image policy network shares the backbone of a "
                    f"Classifier, not of a {type(share_with).__name__}"
                )
                raise TypeError(msg)
            # The fresh backbone is still loaded for its expansion, as
            # DIME always has, so the RNG draws stay the same
            backbone = share_with.backbone
        return ConvNet(backbone, expansion).to(self.device)

    @override
    def check_policy_output(self, policy_network: nn.Module) -> None:
        features, _ = next(iter(self.train_loader))
        features = features.to(self.device)
        with torch.no_grad():
            scores = policy_network(
                self.mask_layer(
                    features,
                    torch.zeros(
                        len(features), self.n_selections, device=self.device
                    ),
                )
            )
        self._check_n_scores(scores.shape[1])

    @override
    def min_lr(
        self, configured: float | None, *, upstream_default: float
    ) -> float:
        del upstream_default
        if configured is None:
            msg = "min_lr must be set for image training"
            raise ValueError(msg)
        return configured

    @override
    def network_pair_spec(self) -> NetworkPairSpec:
        return NetworkPairSpec(
            modality="image",
            d_out=self.d_out,
            backbone=self.architecture.backbone_type,
            image_size=self.image_size,
            patch_size=self.patch_size,
            mask_width=self.mask_width,
        )


def load_backbone(backbone_type: str) -> tuple[nn.Module, int]:
    """Load a ResNet with ImageNet weights, without its head, and its block expansion."""
    if backbone_type == "resnet18":
        base = resnet18(pretrained=True)
    elif backbone_type == "resnet50":
        base = resnet50(pretrained=True)
    else:
        msg = f"Unsupported backbone type: {backbone_type}"
        raise ValueError(msg)
    return ResNet18Backbone(base)


def image_loaders(
    train_dataset: AFADataset,
    val_dataset: AFADataset,
    batch_size: int,
    *,
    smoke_test: bool,
) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Batch images as the dataset yields them: one-hot labels, read lazily."""
    batch_size = training_batch_size(
        smoke_test=smoke_test, default_batch_size=batch_size
    )
    train_loader = DataLoader(
        dataset_subset(train_dataset, smoke_test=smoke_test),  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset_subset(val_dataset, smoke_test=smoke_test),  # pyright: ignore[reportArgumentType]
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader


def prepare_modality(
    cfg: ModalityConfig,
    inputs: FitInputs,
    *,
    classifier_arch: dict[str, Any],
) -> ModalitySetup:
    """Set up the modality of the pretrained model `classifier_arch` describes."""
    classifier_type = classifier_arch["type"]
    if classifier_type in IMAGE_BACKBONES:
        return _prepare_image(cfg, inputs, classifier_arch)
    if classifier_type != "mlp":
        msg = (
            f"No modality for pretrained model type {classifier_type!r}; "
            f"expected 'mlp' or one of {IMAGE_BACKBONES}"
        )
        raise ValueError(msg)
    return _prepare_tabular(cfg, inputs)


def _prepare_tabular(cfg: ModalityConfig, inputs: FitInputs) -> TabularSetup:
    device = torch.device(cfg.device)
    train_dataset = inputs.train_dataset()
    unmasker = inputs.unmasker()
    _, train_labels = train_dataset.get_all_data()
    class_weights = 1 / get_class_frequencies(train_labels)
    class_weights = class_weights / class_weights.sum()
    feature_costs = train_dataset.get_feature_acquisition_costs()
    train_loader, val_loader, d_in, d_out = prepare_datasets(
        train_dataset,
        inputs.val_dataset(),
        cfg.batch_size,
        smoke_test=cfg.smoke_test,
    )
    return TabularSetup(
        device=device,
        architecture=cfg.tabular_architecture,
        train_loader=train_loader,
        val_loader=val_loader,
        d_in=d_in,
        d_out=d_out,
        n_selections=unmasker.get_n_selections(train_dataset.feature_shape),
        mask_layer=MaskLayer(append=True),
        class_weights=class_weights.to(device),
        feature_costs=feature_costs.to(device),
        selection_costs=unmasker.get_selection_costs(feature_costs),
    )


def _prepare_image(
    cfg: ModalityConfig,
    inputs: FitInputs,
    classifier_arch: dict[str, Any],
) -> ImageSetup:
    train_dataset = inputs.train_dataset()
    image_size = int(classifier_arch["image_size"])
    patch_size = int(classifier_arch["patch_size"])
    mask_width = int(classifier_arch["mask_width"])
    if image_size % patch_size != 0:
        msg = (
            f"Pretrained image_size {image_size} is not a multiple of "
            f"its patch_size {patch_size}"
        )
        raise ValueError(msg)
    device = torch.device(cfg.device)
    unmasker = inputs.unmasker()
    # The image Unmasker derives selection costs on CPU; the trainer moves
    # those costs to its device after derivation.
    feature_costs = train_dataset.get_feature_acquisition_costs().cpu()
    n_selections = unmasker.get_n_selections(train_dataset.feature_shape)
    if n_selections != mask_width**2:
        msg = (
            f"The Unmasker offers {n_selections} selections, but the "
            f"pretrained model has {mask_width**2} patches"
        )
        raise ValueError(msg)
    train_loader, val_loader = image_loaders(
        train_dataset,
        inputs.val_dataset(),
        cfg.batch_size,
        smoke_test=cfg.smoke_test,
    )
    return ImageSetup(
        device=device,
        architecture=cfg.image_architecture,
        train_loader=train_loader,
        val_loader=val_loader,
        d_in=train_dataset.feature_shape.numel(),
        d_out=train_dataset.label_shape[0],
        n_selections=n_selections,
        mask_layer=MaskLayer2d(
            mask_width=mask_width, patch_size=patch_size, append=False
        ),
        # Image training keeps its unweighted classification loss.
        class_weights=None,
        feature_costs=feature_costs,
        selection_costs=unmasker.get_selection_costs(feature_costs),
        image_size=image_size,
        patch_size=patch_size,
        mask_width=mask_width,
    )
