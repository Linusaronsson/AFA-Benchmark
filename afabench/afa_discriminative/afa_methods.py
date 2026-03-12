from __future__ import annotations

import logging
import math
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Self, override

import numpy as np
import torch
from torch import nn, optim
from torchrl.modules import MLP
from tqdm.auto import tqdm

from afabench.afa_discriminative.models import (
    ConvNet,
    Predictor,
    NotMIWAE,
    ResNet18Backbone,
    resnet18,
    resnet50,
)
from afabench.afa_discriminative.utils import (
    ConcreteSelector,
    MaskLayer,
    MaskLayer2d,
    get_entropy,
    ind_to_onehot,
    make_onehot,
    to_class_indices,
    patch_soft_to_feature_soft,
    restore_parameters,
    selection_soft_to_feature_soft,
)
from afabench.common.custom_types import (
    AFAAction,
    AFAInitializer,
    AFAMethod,
    AFAUnmasker,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)
from afabench.common.unmaskers import AFAContextUnmasker, CubeNMARUnmasker

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _feature_forbidden_to_selection_forbidden(
    forbidden_feat: FeatureMask,
    *,
    n_selections: int,
    feature_shape: torch.Size,
    unmasker: AFAUnmasker,
) -> SelectionMask:
    """Convert feature-space forbidden masks to selection space when needed."""
    flat_forbidden = forbidden_feat.reshape(forbidden_feat.shape[0], -1)
    if flat_forbidden.shape[1] == n_selections:
        return flat_forbidden

    n_features = math.prod(feature_shape)
    if flat_forbidden.shape[1] != n_features:
        msg = (
            "Forbidden feature mask has incompatible shape. Expected trailing "
            f"dim {n_features} or {n_selections}, got {flat_forbidden.shape[1]}."
        )
        raise ValueError(msg)

    # Special case for AFAContextUnmasker, which has a specific mapping from features to selections.
    # Could probably be unified to a general case.
    if isinstance(unmasker, AFAContextUnmasker):
        n_contexts = unmasker.n_contexts
        sel_forbidden = torch.zeros(
            (flat_forbidden.shape[0], n_selections),
            dtype=torch.bool,
            device=forbidden_feat.device,
        )
        sel_forbidden[:, 0] = flat_forbidden[:, :n_contexts].any(dim=1)
        sel_forbidden[:, 1:] = flat_forbidden[:, n_contexts:]
        return sel_forbidden

    if isinstance(unmasker, CubeNMARUnmasker):
        excluded_start = unmasker.n_contexts
        sel_forbidden = torch.zeros(
            (flat_forbidden.shape[0], n_selections),
            dtype=torch.bool,
            device=forbidden_feat.device,
        )
        sel_forbidden[:, 0] = flat_forbidden[
            :, : unmasker.n_contexts
        ].any(dim=1)
        sel_forbidden[:, 1:] = flat_forbidden[:, excluded_start:]
        return sel_forbidden

    if flat_forbidden.any():
        msg = (
            "Cannot convert a non-empty feature-space forbidden mask to "
            f"selection space for unmasker {type(unmasker).__name__}."
        )
        raise ValueError(msg)

    return torch.zeros(
        (flat_forbidden.shape[0], n_selections),
        dtype=torch.bool,
        device=forbidden_feat.device,
    )


class GreedyDynamicSelection(nn.Module):
    """
    Greedy adaptive feature selection.

    Args:
      selector:
      predictor:
      mask_layer:
      selector_layer:

    """

    def __init__(
        self,
        selector: nn.Module,
        predictor: nn.Module,
        mask_layer: MaskLayer | MaskLayer2d,
        initializer: AFAInitializer,
        unmasker: AFAUnmasker,
    ) -> None:
        super().__init__()

        # Set up models and mask layer.
        self.selector: nn.Module = selector
        self.predictor: nn.Module = predictor
        self.mask_layer: MaskLayer | MaskLayer2d = mask_layer

        # Set up selector layer.
        self.selector_layer: nn.Module = ConcreteSelector()

        self.initializer: AFAInitializer = initializer
        self.unmasker: AFAUnmasker = unmasker

    def fit(  # noqa: PLR0915, PLR0912, C901
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        lr: float,
        nepochs: int,
        max_features: int | None,
        loss_fn: nn.Module,
        val_loss_fn: nn.Module | None = None,
        val_loss_mode: str | None = None,
        factor: float = 0.2,
        patience: int = 2,
        min_lr: float = 1e-5,
        early_stopping_epochs: int | None = None,
        start_temp: float = 1.0,
        end_temp: float = 0.1,
        temp_steps: int = 5,
        argmax: bool = False,  # noqa: FBT002
        verbose: bool = True,  # noqa: FBT002
        feature_costs: torch.Tensor | None = None,
    ) -> None:
        """Train model to perform greedy adaptive feature selection."""
        # Verify arguments.
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = "min"
        elif val_loss_mode is None:
            msg = "must specify val_loss_mode (min or max) when validation_loss_fn is specified"
            raise ValueError(msg)
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1

        # Set up models.
        selector = self.selector
        predictor = self.predictor
        mask_layer = self.mask_layer
        selector_layer = self.selector_layer
        initializer = self.initializer
        unmasker = self.unmasker
        device = next(predictor.parameters()).device
        val_loss_fn.to(device)

        # Determine mask size.
        if mask_layer.mask_size is not None:
            mask_size = int(mask_layer.mask_size)
        else:
            # Must be tabular (1d data).
            x, _ = next(iter(val_loader))
            assert len(x.shape) == 2
            mask_size = x.shape[1]

        x0, _ = next(iter(val_loader))
        x0 = x0.to(device)
        feature_shape = torch.Size(list(x0.shape[1:]))

        if feature_costs is None:
            if len(feature_shape) == 3:
                C, H, W = feature_shape
                feature_costs = torch.ones((C, H, W), device="cpu")
            else:
                feature_costs = torch.ones(mask_size, device="cpu")
        elif isinstance(feature_costs, np.ndarray):
            feature_costs = torch.tensor(feature_costs, device="cpu")
        selection_costs = unmasker.get_selection_costs(feature_costs).to(device)
        log_cost = torch.log(selection_costs)

        n_selections = unmasker.get_n_selections(feature_shape)

        # For tracking best models with zero temperature.
        best_val = None
        best_zerotemp_selector = None
        best_zerotemp_predictor = None

        # Train separately with each temperature.
        total_epochs = 0
        for temp in np.geomspace(start_temp, end_temp, temp_steps):
            if verbose:
                log.info("Starting DIME training with temp=%.4f.", temp)

            # Set up optimizer and lr scheduler.
            opt = optim.Adam(
                set(
                    list(predictor.parameters()) + list(selector.parameters())
                ),
                lr=lr,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode=val_loss_mode,  # pyright: ignore[reportArgumentType]
                factor=factor,
                patience=patience,
                min_lr=min_lr,
            )

            # For tracking best models and early stopping.
            best_selector = deepcopy(selector)
            best_predictor = deepcopy(predictor)
            num_bad_epochs = 0
            epoch = 0
            epoch_iterator = range(nepochs)
            progress_bar = None
            if verbose:
                progress_bar = tqdm(
                    epoch_iterator,
                    desc=f"DIME train T={temp:.4f}",
                    leave=False,
                )
                epoch_iterator = progress_bar

            for _epoch in epoch_iterator:
                # Switch models to training mode.
                selector.train()
                predictor.train()
                epoch_train_loss = 0.0
                for x_batch, y_batch in train_loader:
                    # Move to device.
                    x = x_batch.to(device)
                    y = to_class_indices(y_batch).to(device)

                    m_sel = torch.zeros(
                        len(x), n_selections, dtype=x.dtype, device=device
                    )
                    # Pixel level mask for image data
                    init_mask_bool = initializer.initialize(
                        features=x,
                        label=y,
                        feature_shape=feature_shape,
                    ).to(device)
                    m_feat = init_mask_bool.to(dtype=x.dtype)

                    selector.zero_grad()
                    predictor.zero_grad()

                    for _ in range(max_features):
                        # Evaluate selector model.
                        # x_masked = mask_layer(x, m_feat)
                        if len(x.shape) == 4:
                            # Always set append=False for image data
                            x_masked = x * m_feat
                        else:
                            x_masked = mask_layer(x, m_feat)
                        logits = selector(x_masked).flatten(1)
                        # since not a probability, do exp(logits)/cost <-> logits / log_cost
                        logits_cost = logits - log_cost

                        # Get selections.
                        # soft = selector_layer(logits, temp)
                        soft = selector_layer(logits_cost, temp)
                        if len(x.shape) == 4:
                            soft_feat = patch_soft_to_feature_soft(soft, x)
                        elif isinstance(unmasker, AFAContextUnmasker):
                            soft_feat = selection_soft_to_feature_soft(
                                soft,
                                mask_size=mask_size,
                                n_contexts=unmasker.n_contexts,
                            )
                        else:
                            soft_feat = soft
                        m_soft_feat = torch.maximum(m_feat, soft_feat)

                        # Evaluate predictor model.
                        if len(x.shape) == 4:
                            x_masked = x * m_soft_feat
                        else:
                            x_masked = mask_layer(x, m_soft_feat)
                        pred = predictor(x_masked)

                        # Calculate loss.
                        loss = loss_fn(pred, y)
                        (loss / max_features).backward()
                        epoch_train_loss += loss.item()

                        # Update mask, ensure no repeats.
                        dist = selector_layer(logits_cost - 1e6 * m_sel, 1e-6)
                        sel_idx = torch.argmax(dist, dim=1, keepdim=True)
                        # Zero-based indexing for unmaskers
                        afa_selection = sel_idx.to(torch.long)
                        m_sel = torch.max(
                            m_sel,
                            make_onehot(dist),
                        )
                        m_feat = unmasker.unmask(
                            masked_features=x_masked,
                            feature_mask=m_feat.bool(),
                            features=x,
                            afa_selection=afa_selection,
                            selection_mask=m_sel,
                            feature_shape=feature_shape,
                        ).to(dtype=x.dtype)

                    # Take gradient step.
                    opt.step()

                # avg_train = epoch_train_loss / len(train_loader)

                # Calculate validation loss.
                selector.eval()
                predictor.eval()
                with torch.no_grad():
                    # For mean loss.
                    pred_list = []
                    hard_pred_list = []
                    label_list = []

                    for x_batch, y_batch in val_loader:
                        # Move to device.
                        x = x_batch.to(device)
                        y = to_class_indices(y_batch).to(device)

                        m_sel = torch.zeros(
                            len(x), n_selections, dtype=x.dtype, device=device
                        )
                        init_mask_bool = initializer.initialize(
                            features=x,
                            label=y,
                            feature_shape=feature_shape,
                        ).to(device)
                        m_feat = init_mask_bool.to(dtype=x.dtype)

                        for _ in range(max_features):
                            # Evaluate selector model.
                            if len(x.shape) == 4:
                                x_masked = x * m_feat
                            else:
                                x_masked = mask_layer(x, m_feat)
                            logits = selector(x_masked).flatten(1)
                            logits_cost = logits - log_cost
                            logits_cost = logits_cost - 1e6 * m_sel

                            # Get selections, ensure no repeats.
                            # logits = logits - 1e6 * m
                            if argmax:
                                soft = selector_layer(
                                    logits_cost, temp, deterministic=True
                                )
                            else:
                                soft = selector_layer(logits_cost, temp)
                            if len(x.shape) == 4:
                                soft_feat = patch_soft_to_feature_soft(soft, x)
                            elif isinstance(unmasker, AFAContextUnmasker):
                                soft_feat = selection_soft_to_feature_soft(
                                    soft,
                                    mask_size,
                                    unmasker.n_contexts,
                                )
                            else:
                                soft_feat = soft
                            m_soft_feat = torch.maximum(m_feat, soft_feat)
                            m_sel = torch.max(m_sel, make_onehot(soft))
                            sel_idx = torch.argmax(soft, dim=1, keepdim=True)
                            afa_selection = sel_idx.to(torch.long)
                            m_feat = unmasker.unmask(
                                masked_features=x_masked,
                                feature_mask=m_feat.bool(),
                                features=x,
                                afa_selection=afa_selection,
                                selection_mask=m_sel,
                                feature_shape=feature_shape,
                            ).to(dtype=x.dtype)

                            # Evaluate predictor with soft sample.
                            if len(x.shape) == 4:
                                x_masked = x * m_soft_feat
                            else:
                                x_masked = mask_layer(x, m_soft_feat)
                            pred = predictor(x_masked)

                            # Evaluate predictor with hard sample.
                            if len(x.shape) == 4:
                                x_masked = x * m_feat
                            else:
                                x_masked = mask_layer(x, m_feat)
                            hard_pred = predictor(x_masked)

                            # Append predictions and labels.
                            pred_list.append(pred)
                            hard_pred_list.append(hard_pred)
                            label_list.append(y)

                    # Calculate mean loss.
                    pred = torch.cat(pred_list, 0)
                    hard_pred = torch.cat(hard_pred_list, 0)
                    y = torch.cat(label_list, 0)
                    val_loss = float(val_loss_fn(pred, y).item())
                    val_hard_loss = float(val_loss_fn(hard_pred, y).item())

                if progress_bar is not None:
                    progress_bar.set_postfix(
                        val_loss=f"{val_loss:.4f}",
                        zero_temp_loss=f"{val_hard_loss:.4f}",
                    )

                # Update scheduler.
                scheduler.step(val_loss)

                # Check if best model.
                if val_loss == scheduler.best:
                    best_selector = deepcopy(selector)
                    best_predictor = deepcopy(predictor)
                    num_bad_epochs = 0
                else:
                    num_bad_epochs += 1

                # Check if best model with zero temperature.
                if (
                    (best_val is None)
                    or (val_loss_mode == "min" and val_hard_loss < best_val)
                    or (val_loss_mode == "max" and val_hard_loss > best_val)
                ):
                    best_val = val_hard_loss
                    best_zerotemp_selector = deepcopy(selector)
                    best_zerotemp_predictor = deepcopy(predictor)

                # Early stopping.
                if num_bad_epochs > early_stopping_epochs:
                    break

            # Update total epoch count.
            if verbose:
                log.info(
                    "Stopping DIME training with temp=%.4f at epoch %d.",
                    temp,
                    epoch + 1,
                )
            total_epochs += epoch + 1

            # Copy parameters from best model.
            restore_parameters(selector, best_selector)
            restore_parameters(predictor, best_predictor)

        # Copy parameters from best model with zero temperature.
        assert best_zerotemp_selector is not None
        assert best_zerotemp_predictor is not None
        restore_parameters(selector, best_zerotemp_selector)
        restore_parameters(predictor, best_zerotemp_predictor)


class Covert2023AFAMethod(AFAMethod):
    def __init__(
        self,
        selector: nn.Module,
        predictor: nn.Module,
        device: torch.device,
        lambda_threshold: float | None = None,
        selection_costs: torch.Tensor | None = None,
        selector_hidden_layers: list[int] | None = None,
        predictor_hidden_layers: list[int] | None = None,
        dropout: float = 0.3,
        modality: str | None = "tabular",
        n_patches: int | None = None,
        d_in: int | None = None,
        d_out: int | None = None,
        n_selections: int | None = None,
        backbone_type: str = "resnet50",
    ):
        super().__init__()

        # Set up models and mask layer.
        self.selector: nn.Module = selector
        self.predictor: nn.Module = predictor
        self._device: torch.device = device
        if lambda_threshold is None:
            self.lambda_threshold: float = -math.inf
        else:
            self.lambda_threshold = lambda_threshold
        self._selection_costs: torch.Tensor | None = selection_costs
        self.selector_hidden_layers = (
            [128, 128]
            if selector_hidden_layers is None
            else selector_hidden_layers
        )
        self.predictor_hidden_layers = (
            [128, 128]
            if predictor_hidden_layers is None
            else predictor_hidden_layers
        )
        self.dropout = dropout
        self.modality: str | None = modality
        # for image selection
        self.n_patches: int | None = n_patches
        self.d_in: int | None = d_in
        self.d_out: int | None = d_out
        self.n_selections: int | None = n_selections
        self.image_size: int | None = None
        self.patch_size: int | None = None
        self.mask_width: int | None = None
        self.backbone_type: str = backbone_type

    def _flat_mask_to_patch_mask(
        self, feature_mask: torch.Tensor
    ) -> torch.Tensor:
        assert feature_mask.dim() == 4
        B, C, H, W = feature_mask.shape
        ps = self.patch_size
        assert ps is not None
        ph = H // ps
        pw = W // ps
        fm = feature_mask.view(B, C, ph, ps, pw, ps)
        patch_revealed = fm.any(dim=(1, 3, 5))
        return patch_revealed.reshape(B, ph * pw)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        if self.modality == "tabular":
            x_masked = torch.cat([masked_features, feature_mask], dim=1)
            pred = self.predictor(x_masked)
        else:
            pred = self.predictor(masked_features)
        return pred.softmax(dim=-1)

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        with torch.no_grad():
            if self.modality == "tabular":
                x_masked_pred = torch.cat([masked_features, feature_mask], dim=1)
                pred = self.predictor(x_masked_pred)
            else:
                pred = self.predictor(masked_features)

            entropy = get_entropy(pred)
            stop_mask = entropy < self.lambda_threshold

        if self.modality == "tabular":
            x_masked = torch.cat([masked_features, feature_mask], dim=1)
            logits = self.selector(x_masked).flatten(1)
            # TODO: currently assume that if we use AFAContextUnmasker, then we have not None selection mask
            if selection_mask is not None:
                assert logits.shape == selection_mask.shape, (
                    f"selection_mask shape {selection_mask.shape} incompatible with logits {logits.shape}"
                )
                logits = logits - 1e6 * selection_mask.float()
            else:
                assert logits.shape == feature_mask.shape, (
                    f"feature_mask shape {feature_mask.shape} incompatible with logits {logits.shape}"
                )
                logits = logits - 1e6 * feature_mask
        else:
            logits = self.selector(masked_features)
            assert logits.dim() == 2, (
                f"Selector must return [B, N], got {logits.shape}"
            )
            patch_mask = self._flat_mask_to_patch_mask(feature_mask).float()
            logits = logits - 1e6 * patch_mask

        if self._selection_costs is not None:
            costs = self._selection_costs.to(self._device)
            costs = torch.clamp(costs, min=1e-12)
            scores = logits / costs.unsqueeze(0)
        else:
            scores = logits
        best_scores, best_idx = scores.max(dim=1)

        # stop_mask = best_scores < lam
        # all masked
        stop_mask = stop_mask | (best_scores < -1e5)

        selections = (best_idx + 1).to(dtype=torch.long).unsqueeze(-1)
        stop_mask = stop_mask.unsqueeze(-1)
        # 0 = stop
        selections = selections.masked_fill(stop_mask, 0)
        return selections

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(path / "model.pt", weights_only=False, map_location=device)
        arch = checkpoint["architecture"]
        lambda_threshold = checkpoint.get("lambda_threshold", None)
        selection_costs = checkpoint.get("selection_costs", None)
        feature_costs = checkpoint.get("feature_costs", None)
        if selection_costs is not None:
            selection_costs = selection_costs.to(device)
        elif feature_costs is not None:
            selection_costs = feature_costs.to(device)
        # tabular
        if arch["type"] == "mlp":
            d_in = arch["d_in"]
            d_out = arch["d_out"]
            n_selections = arch.get("n_selections", None)
            if n_selections is None:
                n_selections = d_in
            selector_hidden_layers = arch["selector_hidden_layers"]
            predictor_hidden_layers = arch["predictor_hidden_layers"]
            dropout = arch["dropout"]
            predictor = MLP(
                in_features=d_in * 2,
                out_features=d_out,
                num_cells=predictor_hidden_layers,
                activation_class=nn.ReLU,
                dropout=dropout,
            )
            selector = MLP(
                in_features=d_in * 2,
                out_features=n_selections,
                num_cells=selector_hidden_layers,
                activation_class=nn.ReLU,
                dropout=dropout,
            )

            model = cls(
                selector=selector,
                predictor=predictor,
                device=device,
                lambda_threshold=lambda_threshold,
                selection_costs=selection_costs,
                selector_hidden_layers=selector_hidden_layers,
                predictor_hidden_layers=predictor_hidden_layers,
                dropout=dropout,
                modality="tabular",
                d_in=d_in,
                d_out=d_out,
                n_selections=n_selections,
            )
            model.selector.load_state_dict(checkpoint["selector_state_dict"])
            model.predictor.load_state_dict(checkpoint["predictor_state_dict"])
            model.selector.eval()
            model.predictor.eval()
            return model.to(device)

        if arch["type"] in ("resnet18", "resnet50"):
            d_out = arch["d_out"]
            if arch["type"] == "resnet18":
                base = resnet18(pretrained=False)
            else:
                base = resnet50(pretrained=False)
            backbone_net, expansion = ResNet18Backbone(base)
            predictor = Predictor(backbone_net, expansion, d_out)
            selector = ConvNet(backbone_net, expansion, 0.5)

            model = cls(
                selector=selector,
                predictor=predictor,
                device=device,
                lambda_threshold=lambda_threshold,
                selection_costs=selection_costs,
                modality="image",
                n_patches=int(arch["mask_width"]) ** 2,
                d_out=d_out,
                backbone_type=str(arch["type"]),
            )

            model.mask_width = int(arch["mask_width"])
            model.patch_size = int(arch["patch_size"])
            model.image_size = int(arch["image_size"])

            model.selector.load_state_dict(checkpoint["selector_state_dict"])
            model.predictor.load_state_dict(checkpoint["predictor_state_dict"])
            model.selector.eval()
            model.predictor.eval()
            return model.to(device)
        msg = "Unrecognized checkpoint format"
        raise ValueError(msg)

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self.modality == "tabular":
            arch = {
                "type": "mlp",
                "d_in": self.d_in,
                "d_out": self.d_out,
                "n_selections": self.n_selections,
                "selector_hidden_layers": self.selector_hidden_layers,
                "predictor_hidden_layers": self.predictor_hidden_layers,
                "dropout": self.dropout,
                "model_type": "tabular",
            }
        else:
            backbone_type = self.backbone_type
            arch = {
                "type": backbone_type,
                "backbone": backbone_type,
                "image_size": getattr(self, "image_size", 224),
                "patch_size": getattr(self, "patch_size", 16),
                "mask_width": getattr(self, "mask_width", 14),
                "d_out": self.d_out,
                "model_type": "image",
            }
        payload = {
            "selector_state_dict": self.selector.state_dict(),
            "predictor_state_dict": self.predictor.state_dict(),
            "architecture": arch,
            "lambda_threshold": float(self.lambda_threshold),
            "selection_costs": self._selection_costs.detach().cpu() if self._selection_costs is not None else None
        }
        torch.save(payload, Path(path) / "model.pt")

    @override
    def to(self, device: torch.device) -> Self:
        self.selector = self.selector.to(device)
        self.predictor = self.predictor.to(device)
        self._device = device
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @property
    @override
    def cost_param(self) -> float | None:
        return float(self.lambda_threshold)

    @override
    def set_cost_param(self, cost_param: float) -> None:
        self.lambda_threshold = cost_param


class CMIEstimator(nn.Module):
    """Greedy CMI estimation module."""

    def __init__(
        self,
        value_network: nn.Module,
        predictor: nn.Module,
        mask_layer: MaskLayer | MaskLayer2d,
        initializer: AFAInitializer,
        unmasker: AFAUnmasker,
        notmiwae_model: NotMIWAE | None = None,
    ):
        super().__init__()

        # Save network modules.
        self.value_network: nn.Module = value_network
        self.predictor: nn.Module = predictor
        self.mask_layer: MaskLayer | MaskLayer2d = mask_layer
        self.initializer: AFAInitializer = initializer
        self.unmasker: AFAUnmasker = unmasker
        self.notmiwae_model: NotMIWAE | None = notmiwae_model

    def _ipw_weight_from_propensity(
        self,
        ipw_normalize_weights: bool,
        ipw_min_propensity: float,
        ipw_max_weight: float,
        propensity_for_actions: torch.Tensor
    ) -> torch.Tensor:
        weights = torch.reciprocal(
            torch.clamp(
                propensity_for_actions,
                min=ipw_min_propensity,
            )
        )
        weights = torch.clamp(weights, max=ipw_max_weight)
        if ipw_normalize_weights:
            weights = weights / torch.clamp(weights.mean(), min=1e-12)
        return weights.detach()

    def fit(  # noqa: PLR0915, PLR0912, C901
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        lr: float,
        nepochs: int,
        max_features: int | None,
        eps: float,
        loss_fn: nn.Module,
        val_loss_fn: nn.Module | None,
        val_loss_mode: str | None,
        factor: float = 0.2,
        patience: int = 2,
        min_lr: float = 1e-6,
        early_stopping_epochs: int | None = None,
        eps_decay: float = 0.2,
        eps_steps: int = 1,
        feature_costs: torch.Tensor | None = None,
        cmi_scaling: str = "bounded",
        ipw_mode: str = "none",
        ipw_min_propensity: float = 1e-3,
        ipw_max_weight: float = 10.0,
        ipw_normalize_weights: bool = True,  # noqa: FBT002
        verbose: bool = True,  # noqa: FBT002
    ) -> None:
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = "min"
        elif val_loss_mode is None:
            msg = "must specify val_loss_mode (min or max) when validation_loss_fn is specified"
            raise ValueError(msg)
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        if ipw_mode not in {"none", "feature_marginal", "notmiwae_feature"}:
            msg = (
                "ipw_mode must be one of {'none', 'feature_marginal', 'notmiwae_feature'}, "
                f"got {ipw_mode}."
            )
            raise ValueError(msg)
        if ipw_mode == "notmiwae_feature" and self.notmiwae_model is None:
            msg = (
                "ipw_mode='notmiwae_feature' requires a trained notmiwae_model."
            )
            raise ValueError(msg)
        if ipw_min_propensity <= 0.0:
            msg = (
                "ipw_min_propensity must be > 0, "
                f"got {ipw_min_propensity}."
            )
            raise ValueError(msg)
        if ipw_max_weight <= 0.0:
            msg = (
                "ipw_max_weight must be > 0, "
                f"got {ipw_max_weight}."
            )
            raise ValueError(msg)
        if max_features is None:
            msg = (
                "CMIEstimator.fit requires an integer max_features. "
                "Soft-budget workflow runs must provide a training hard "
                "budget for discriminative methods."
            )
            raise ValueError(msg)

        value_network: nn.Module = self.value_network
        predictor: nn.Module = self.predictor
        mask_layer: MaskLayer | MaskLayer2d = self.mask_layer
        initializer: AFAInitializer = self.initializer
        unmasker: AFAUnmasker = self.unmasker

        device = next(predictor.parameters()).device
        val_loss_fn = val_loss_fn.to(device)
        value_network = value_network.to(device)

        if mask_layer.mask_size is not None:
            mask_size = int(mask_layer.mask_size)
        else:
            # Must be tabular (1d data).
            x, y = next(iter(val_loader))
            assert len(x.shape) == 2
            mask_size = x.shape[1]

        x0, _ = next(iter(val_loader))
        x0 = x0.to(device)
        feature_shape = torch.Size(list(x0.shape[1:]))

        if feature_costs is None:
            if len(feature_shape) == 3:
                C, H, W = feature_shape
                feature_costs = torch.ones((C, H, W), device="cpu")
            else:
                feature_costs = torch.ones(mask_size, device="cpu")
        elif isinstance(feature_costs, np.ndarray):
            feature_costs = torch.tensor(feature_costs).to("cpu")
        selection_costs = unmasker.get_selection_costs(feature_costs).to(device)
        selection_costs = torch.clamp(selection_costs, min=1e-12)

        n_selections = unmasker.get_n_selections(feature_shape)

        if ipw_mode == "notmiwae_feature" and n_selections != mask_size:
            msg = (
                "ipw_mode='notmiwae_feature' currently requires per feature selection "
                f"(got n_selections={n_selections}, mask_size={mask_size})."
            )
            raise ValueError(msg)

        opt = optim.Adam(
            set(
                list(value_network.parameters()) + list(predictor.parameters())
            ),
            lr=lr,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=val_loss_mode,  # pyright: ignore[reportArgumentType]
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

        # For tracking best models and early stopping.
        best_value_network = deepcopy(value_network)
        best_predictor = deepcopy(predictor)
        num_bad_epochs = 0
        num_epsilon_steps = 0

        if verbose:
            log.info(
                "Training CMI estimator for %d epochs with max_features=%s and initial eps=%.5f.",
                nepochs,
                max_features,
                eps,
            )

        epoch_iterator = tqdm(
            range(nepochs),
            total=nepochs,
            desc="Training CMI estimator",
            disable=not verbose,
        )
        for _epoch in epoch_iterator:
            # Switch models to training mode.
            value_network.train()
            predictor.train()
            value_losses = []
            pred_losses = []
            total_loss = 0

            for x_batch, y_batch in train_loader:
                # Move to device.
                x = x_batch.to(device)
                y = to_class_indices(y_batch).to(device)
                batch_idx = torch.arange(len(x), device=device)

                init_mask_bool = initializer.initialize(
                    features=x,
                    label=y,
                    feature_shape=feature_shape,
                ).to(device)
                forbidden_feat = initializer.get_training_forbidden_mask(
                    init_mask_bool
                ).to(device)
                # init_mask_bool = init_mask_bool & ~forbidden_feat
                # m_feat = init_mask_bool.to(dtype=x.dtype)
                m_feat = torch.zeros_like(init_mask_bool, dtype=x.dtype)

                forbidden_sel = _feature_forbidden_to_selection_forbidden(
                    forbidden_feat,
                    n_selections=n_selections,
                    feature_shape=feature_shape,
                    unmasker=unmasker,
                )

                m_sel = torch.zeros(
                    len(x), n_selections, dtype=x.dtype, device=device
                )
                m_sel = torch.maximum(m_sel, forbidden_sel.to(dtype=x.dtype))
                notmiwae_propensity = None
                if ipw_mode == "notmiwae_feature":
                    assert self.notmiwae_model is not None
                    obs_mask = (~forbidden_feat).to(dtype=x.dtype)
                    x_obs = x * obs_mask
                    self.notmiwae_model.eval()
                    with torch.no_grad():
                        feature_probs = self.notmiwae_model.feature_observation_probs(
                            x_filled=x_obs,
                            s=obs_mask,
                            n_samples=100,
                        )
                        feature_probs = feature_probs.masked_fill(
                            forbidden_feat, 0.0
                        )
                        notmiwae_propensity = feature_probs

                value_network.zero_grad()
                predictor.zero_grad()
                value_network_loss_total = 0
                pred_loss_total = 0

                # Predictor loss with initial features.
                if len(x.shape) == 4:
                    x_masked = x * m_feat
                else:
                    x_masked = mask_layer(x, m_feat)
                pred_without_next_feature = predictor(x_masked)
                loss_without_next_feature = loss_fn(
                    pred_without_next_feature, y
                )
                pred_loss = loss_without_next_feature.mean()
                pred_loss_total += pred_loss.detach()

                (pred_loss / (max_features + 1)).backward()
                pred_without_next_feature = pred_without_next_feature.detach()
                loss_without_next_feature = loss_without_next_feature.detach()

                for _ in range(max_features):
                    # Estimate CMI using value network.
                    if len(x.shape) == 4:
                        x_masked = x * m_feat
                    else:
                        x_masked = mask_layer(x, m_feat)
                    if cmi_scaling == "bounded":
                        entropy = get_entropy(
                            pred_without_next_feature
                        ).unsqueeze(1)
                        pred_cmi = value_network(x_masked).sigmoid() * entropy
                    elif cmi_scaling == "positive":
                        pred_cmi = torch.nn.functional.softplus(
                            value_network(x_masked)
                        )
                    else:
                        pred_cmi = value_network(x_masked)

                    pred_cmi = pred_cmi.masked_fill(forbidden_sel, -1e9)

                    best = torch.argmax(pred_cmi / selection_costs, dim=1)
                    # rng = np.random.default_rng()
                    allowed = ~forbidden_sel
                    w = allowed.to(torch.float32)
                    # random = torch.tensor(
                    #     np.random.choice(n_selections, size=len(x)),
                    #     device=x.device,
                    # )
                    random = torch.multinomial(w, num_samples=1).squeeze(1).to(x.device)
                    exploit = (torch.rand(len(x), device=x.device) > eps).int()
                    actions = exploit * best + (1 - exploit) * random
                    afa_selection = actions.to(torch.long)
                    afa_selection = afa_selection.unsqueeze(1)
                    m_sel = torch.max(m_sel, ind_to_onehot(actions, n_selections))

                    # Predictor loss.
                    m_feat = unmasker.unmask(
                        masked_features=x_masked,
                        feature_mask=m_feat.bool(),
                        features=x,
                        afa_selection=afa_selection,
                        selection_mask=m_sel,
                        feature_shape=feature_shape,
                    )
                    if len(x.shape) == 4:
                        x_masked = x * m_feat
                    else:
                        x_masked = self.mask_layer(x, m_feat)
                    pred_with_next_feature = predictor(x_masked)
                    loss_with_next_feature = loss_fn(pred_with_next_feature, y)

                    # Value network loss.
                    delta = (
                        loss_without_next_feature
                        - loss_with_next_feature.detach()
                    )
                    pred_selected = pred_cmi[torch.arange(len(x)), actions]
                    squared_error = torch.square(pred_selected - delta)
                    if ipw_mode == "none":
                        value_network_loss = squared_error.mean()
                    elif ipw_mode == "feature_marginal":
                        available_sel = (~forbidden_sel).to(dtype=x.dtype)
                        propensity = available_sel.mean(dim=0)
                        propensity_for_actions = propensity[actions]
                        weights = self._ipw_weight_from_propensity(
                            ipw_normalize_weights=ipw_normalize_weights,
                            ipw_min_propensity=ipw_min_propensity,
                            ipw_max_weight=ipw_max_weight,
                            propensity_for_actions=propensity_for_actions,
                        )
                        value_network_loss = (
                            weights * squared_error
                        ).mean()
                    elif ipw_mode == "notmiwae_feature":
                        assert notmiwae_propensity is not None
                        propensity_for_actions = notmiwae_propensity[batch_idx, actions]
                        weights = self._ipw_weight_from_propensity(
                            ipw_normalize_weights=ipw_normalize_weights,
                            ipw_min_propensity=ipw_min_propensity,
                            ipw_max_weight=ipw_max_weight,
                            propensity_for_actions=propensity_for_actions,
                        )
                        value_network_loss = (
                            weights * squared_error
                        ).mean()
                    else:
                        msg = f"Unsupported ipw_mode: {ipw_mode}"
                        raise RuntimeError(msg)

                    # Calculate gradients.
                    total_loss = torch.mean(value_network_loss) + torch.mean(
                        loss_with_next_feature
                    )
                    (total_loss / (max_features + 1)).backward()

                    # Updates.
                    value_network_loss_total += torch.mean(value_network_loss)
                    pred_loss_total += torch.mean(loss_with_next_feature)
                    loss_without_next_feature = loss_with_next_feature.detach()
                    pred_without_next_feature = pred_with_next_feature.detach()

                # Take gradient step.
                opt.step()
                opt.zero_grad()

                value_losses.append(value_network_loss_total / max_features)
                pred_losses.append(pred_loss_total / (max_features + 1))

            # Calculate validation loss.
            value_network.eval()
            predictor.eval()
            val_preds = [[] for _ in range(max_features + 1)]
            val_targets = []

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    # Move to device.
                    x = x_batch.to(device)
                    y = to_class_indices(y_batch).to(device)

                    # Setup.
                    init_mask_bool = initializer.initialize(
                        features=x,
                        label=y,
                        feature_shape=feature_shape,
                    ).to(device)
                    forbidden_feat = initializer.get_training_forbidden_mask(
                        init_mask_bool
                    ).to(device)
                    forbidden_sel = _feature_forbidden_to_selection_forbidden(
                        forbidden_feat,
                        n_selections=n_selections,
                        feature_shape=feature_shape,
                        unmasker=unmasker,
                    )
                    m_sel = torch.zeros(
                        len(x), n_selections, dtype=x.dtype, device=device
                    )
                    m_sel = torch.maximum(m_sel, forbidden_sel.to(dtype=x.dtype))

                    # init_mask_bool = init_mask_bool & ~forbidden_feat
                    # m_feat = init_mask_bool.to(dtype=x.dtype)
                    m_feat = torch.zeros_like(init_mask_bool, dtype=x.dtype)
                    if len(x.shape) == 4:
                        x_masked = x * m_feat
                    else:
                        x_masked = self.mask_layer(x, m_feat)
                    pred = predictor(x_masked)
                    val_preds[0].append(pred)

                    for i in range(1, max_features + 1):
                        # Estimate CMI using value network.
                        if len(x.shape) == 4:
                            x_masked = x * m_feat
                        else:
                            x_masked = mask_layer(x, m_feat)
                        if cmi_scaling == "bounded":
                            entropy = get_entropy(pred).unsqueeze(1)
                            pred_cmi = (
                                value_network(x_masked).sigmoid() * entropy
                            )
                        elif cmi_scaling == "positive":
                            pred_cmi = torch.nn.functional.softplus(
                                value_network(x_masked)
                            )
                        else:
                            pred_cmi = value_network(x_masked)

                        # Select next feature, ensure no repeats.
                        pred_cmi -= 1e6 * m_sel
                        best_feature_index = torch.argmax(
                            pred_cmi / selection_costs, dim=1
                        )
                        m_sel = torch.max(
                            m_sel, ind_to_onehot(best_feature_index, n_selections)
                        )
                        afa_selection = best_feature_index.to(torch.long)
                        afa_selection = afa_selection.unsqueeze(1)
                        m_feat = unmasker.unmask(
                            masked_features=x_masked,
                            feature_mask=m_feat.bool(),
                            features=x,
                            afa_selection=afa_selection,
                            selection_mask=m_sel,
                            feature_shape=feature_shape,
                        )

                        # Make prediction.
                        if len(x.shape) == 4:
                            x_masked = x * m_feat
                        else:
                            x_masked = self.mask_layer(x, m_feat)
                        pred = self.predictor(x_masked)
                        val_preds[i].append(pred)

                    val_targets.append(y)

                # Calculate mean loss.
                y_val = torch.cat(val_targets)
                preds_cat = [torch.cat(p) for p in val_preds]
                pred_losses = [loss_fn(p, y_val).mean() for p in preds_cat]
                val_scores = [val_loss_fn(p, y_val) for p in preds_cat]
                val_loss_mean = torch.stack(pred_losses).mean()
                val_perf_mean = torch.stack(val_scores).mean()
                val_loss_final = pred_losses[-1]
                val_perf_final = val_scores[-1]

            # log_payload = {
            #     "cmi_estimator/train_loss": total_loss / (max_features + 1),
            # }
            # if user_supplied_val_metric:
            #     log_payload["cmi_estimator/val_accuracy"] = val_perf_mean
            # else:
            #     log_payload["cmi_estimator/val_loss"] = val_loss_mean
            # wandb.log(
            #     {
            #         "cmi_estimator/train_loss": total_loss
            #         / (max_features + 1),
            #         "cmi_estimator/val_loss": val_loss_mean,
            #         "cmi_estimator/val_accuracy": val_perf_mean,
            #     }
            # )
            # wandb.log(log_payload)

            val_perf_final_display = float(val_perf_final.mean().item())
            epoch_iterator.set_postfix(
                val_loss_mean=f"{val_loss_mean.item():.4f}",
                val_perf_mean=f"{val_perf_mean.item():.4f}",
                val_loss_final=f"{val_loss_final.item():.4f}",
                val_perf_final=f"{val_perf_final_display:.4f}",
                eps=f"{eps:.4f}",
            )

            # Update scheduler.
            scheduler.step(val_perf_mean)

            # Check if best model.
            if val_perf_mean == scheduler.best:
                best_value_network = deepcopy(value_network)
                best_predictor = deepcopy(predictor)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            # Decay epsilon.
            if num_bad_epochs > early_stopping_epochs:
                eps = eps * eps_decay
                num_bad_epochs = 0
                num_epsilon_steps += 1
                log.info(
                    "Decayed CMI estimator eps to %.5f (step %d/%d).",
                    eps,
                    num_epsilon_steps,
                    eps_steps,
                )

                # Early stopping.
                if num_epsilon_steps >= eps_steps:
                    log.info(
                        "Stopping CMI estimator after reaching %d epsilon decay steps.",
                        num_epsilon_steps,
                    )
                    break

                # Reset optimizer learning rate. Could fully reset optimizer and scheduler, but this is simpler.
                for g in opt.param_groups:
                    g["lr"] = lr

        # Copy parameters from best model.
        restore_parameters(value_network, best_value_network)
        restore_parameters(predictor, best_predictor)


class Gadgil2023AFAMethod(AFAMethod):
    def __init__(
        self,
        value_network: nn.Module,
        predictor: nn.Module,
        device: torch.device,
        lambda_threshold: float | None = None,
        selection_costs: torch.Tensor | None = None,
        value_network_hidden_layers: list[int] | None = None,
        predictor_hidden_layers: list[int] | None = None,
        dropout: float = 0.3,
        modality: str | None = "tabular",
        n_patches: int | None = None,
        d_in: int | None = None,
        d_out: int | None = None,
        n_selections: int | None = None,
        backbone_type: str = "resnet50",
    ):
        super().__init__()

        # Save network modules.
        self.value_network: nn.Module = value_network
        self.predictor: nn.Module = predictor
        self._device: torch.device = device
        if lambda_threshold is None:
            self.lambda_threshold: float = -math.inf
        else:
            self.lambda_threshold = lambda_threshold
        self._selection_costs: torch.Tensor | None = selection_costs
        self.value_network_hidden_layers = (
            [128, 128]
            if value_network_hidden_layers is None
            else value_network_hidden_layers
        )
        self.predictor_hidden_layers = (
            [128, 128]
            if predictor_hidden_layers is None
            else predictor_hidden_layers
        )
        self.dropout = dropout
        self.modality: str | None = modality
        self.n_patches: int | None = n_patches
        self.d_in: int | None = d_in
        self.d_out: int | None = d_out
        self.n_selections: int | None = n_selections
        self.image_size: int | None = None
        self.patch_size: int | None = None
        self.mask_width: int | None = None
        self.backbone_type: str = backbone_type

        # Per-feature CMI logging (disabled by default).
        self._log_cmi: bool = False
        self._cmi_log: list[dict[str, torch.Tensor]] = []

    def _flat_mask_to_patch_mask(
        self, feature_mask: torch.Tensor
    ) -> torch.Tensor:
        # need to check, ph, pw, which comes first?
        assert feature_mask.dim() == 4
        B, C, H, W = feature_mask.shape
        ps = self.patch_size
        assert ps is not None
        ph = H // ps
        pw = W // ps
        fm = feature_mask.view(B, C, ph, ps, pw, ps)
        patch_revealed = fm.any(dim=(1, 3, 5))
        return patch_revealed.reshape(B, ph * pw)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        if self.modality == "tabular":
            x_masked = torch.cat([masked_features, feature_mask], dim=1)
            pred = self.predictor(x_masked)
        else:
            pred = self.predictor(masked_features)
        return pred.softmax(dim=-1)

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        if self.modality == "tabular":
            x_masked = torch.cat([masked_features, feature_mask], dim=1)
            # pred = self.predict(masked_features, feature_mask)
            pred = self.predictor(x_masked)
            entropy = get_entropy(pred).unsqueeze(1)
            pred_cmi = self.value_network(x_masked).sigmoid() * entropy
            if selection_mask is not None:
                assert pred_cmi.shape == selection_mask.shape, (
                    f"selection_mask shape {selection_mask.shape} incompatible with pred_cmi {pred_cmi.shape}"
                )
                pred_cmi -= 1e6 * selection_mask.float()
            else:
                assert pred_cmi.shape == feature_mask.shape, (
                    f"feature_mask shape {feature_mask.shape} incompatible with pred_cmi {pred_cmi.shape}"
                )
                pred_cmi -= 1e6 * feature_mask
        else:
            pred = self.predictor(masked_features)
            entropy = get_entropy(pred).unsqueeze(1)
            pred_cmi = self.value_network(masked_features).sigmoid() * entropy
            patch_mask = self._flat_mask_to_patch_mask(feature_mask).float()
            pred_cmi = pred_cmi - 1e6 * patch_mask

        if self._selection_costs is not None:
            costs = self._selection_costs.to(self._device)
            costs = torch.clamp(costs, min=1e-12)
            scores = pred_cmi / costs.unsqueeze(0)
        else:
            scores = pred_cmi
        best_scores, best_idx = scores.max(dim=1)
        lam = self.lambda_threshold
        stop_mask = best_scores < lam
        stop_mask = stop_mask | (best_scores < -1e5)

        selections = (best_idx + 1).to(dtype=torch.long).unsqueeze(-1)
        stop_mask = stop_mask.unsqueeze(-1)
        selections = selections.masked_fill(stop_mask, 0)

        if self._log_cmi:
            self._cmi_log.append({
                "pred_cmi": pred_cmi.detach().cpu(),
                "entropy": entropy.detach().cpu(),
                "feature_mask": feature_mask.detach().cpu(),
                "selected_action": selections.detach().cpu(),
            })

        return selections

    def enable_cmi_logging(self) -> None:
        """Enable per-feature CMI score logging during act()."""
        self._log_cmi = True

    def disable_cmi_logging(self) -> None:
        """Disable per-feature CMI score logging."""
        self._log_cmi = False

    def get_cmi_log(self) -> list[dict[str, torch.Tensor]]:
        """Return the accumulated CMI log entries."""
        return self._cmi_log

    def clear_cmi_log(self) -> None:
        """Clear the accumulated CMI log."""
        self._cmi_log = []

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(path / "model.pt", weights_only=False, map_location=device)
        arch = checkpoint["architecture"]
        lambda_threshold = checkpoint.get("lambda_threshold", None)
        selection_costs = checkpoint.get("selection_costs", None)
        feature_costs = checkpoint.get("feature_costs", None)
        if selection_costs is not None:
            selection_costs = selection_costs.to(device)
        elif feature_costs is not None:
            selection_costs = feature_costs.to(device)
        if arch["type"] == "mlp":
            d_in = arch["d_in"]
            d_out = arch["d_out"]
            n_selections = arch.get("n_selections", None)
            if n_selections is None:
                n_selections = d_in
            value_network_hidden_layers = arch["value_network_hidden_layers"]
            predictor_hidden_layers = arch["predictor_hidden_layers"]
            dropout = arch["dropout"]
            predictor = MLP(
                in_features=d_in * 2,
                out_features=d_out,
                num_cells=predictor_hidden_layers,
                activation_class=nn.ReLU,
                dropout=dropout,
            )
            value_network = MLP(
                in_features=d_in * 2,
                out_features=n_selections,
                num_cells=value_network_hidden_layers,
                activation_class=nn.ReLU,
                dropout=dropout,
            )

            model = cls(
                value_network=value_network,
                predictor=predictor,
                device=device,
                lambda_threshold=lambda_threshold,
                selection_costs=selection_costs,
                value_network_hidden_layers=value_network_hidden_layers,
                predictor_hidden_layers=predictor_hidden_layers,
                dropout=dropout,
                modality="tabular",
                d_in=d_in,
                d_out=d_out,
                n_selections=n_selections,
            )
            model.value_network.load_state_dict(
                checkpoint["value_network_state_dict"]
            )
            model.predictor.load_state_dict(checkpoint["predictor_state_dict"])
            model.value_network.eval()
            model.predictor.eval()
            return model.to(device)

        if arch["type"] in ("resnet18", "resnet50"):
            d_out = arch["d_out"]
            if arch["type"] == "resnet18":
                base = resnet18(pretrained=False)
            else:
                base = resnet50(pretrained=False)
            backbone_net, expansion = ResNet18Backbone(base)
            predictor = Predictor(backbone_net, expansion, d_out)
            value_network = ConvNet(backbone_net, expansion, 0.5)

            model = cls(
                value_network=value_network,
                predictor=predictor,
                device=device,
                lambda_threshold=lambda_threshold,
                selection_costs=selection_costs,
                modality="image",
                n_patches=int(arch["mask_width"]) ** 2,
                d_out=d_out,
                backbone_type=str(arch["type"]),
            )
            model.mask_width = int(arch["mask_width"])
            model.patch_size = int(arch["patch_size"])
            model.image_size = int(arch["image_size"])

            model.value_network.load_state_dict(
                checkpoint["value_network_state_dict"]
            )
            model.predictor.load_state_dict(checkpoint["predictor_state_dict"])
            model.value_network.eval()
            model.predictor.eval()
            return model.to(device)
        msg = "Unrecognized checkpoint format"
        raise ValueError(msg)

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self.modality == "tabular":
            arch = {
                "type": "mlp",
                "d_in": self.d_in,
                "d_out": self.d_out,
                "n_selections": self.n_selections,
                "value_network_hidden_layers": self.value_network_hidden_layers,
                "predictor_hidden_layers": self.predictor_hidden_layers,
                "dropout": self.dropout,
                "model_type": "tabular",
            }
        else:
            backbone_type = self.backbone_type
            arch = {
                "type": backbone_type,
                "backbone": backbone_type,
                "image_size": getattr(self, "image_size", 224),
                "patch_size": getattr(self, "patch_size", 16),
                "mask_width": getattr(self, "mask_width", 14),
                "d_out": self.d_out,
                "model_type": "image",
            }
        payload = {
            "value_network_state_dict": self.value_network.state_dict(),
            "predictor_state_dict": self.predictor.state_dict(),
            "architecture": arch,
            "lambda_threshold": float(self.lambda_threshold),
            "selection_costs": self._selection_costs.detach().cpu() if self._selection_costs is not None else None
        }
        torch.save(payload, Path(path) / "model.pt")

    @override
    def to(self, device: torch.device) -> Self:
        self.value_network = self.value_network.to(device)
        self.predictor = self.predictor.to(device)
        self._device = device
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return True

    @property
    @override
    def cost_param(self) -> float | None:
        return float(self.lambda_threshold)

    @override
    def set_cost_param(self, cost_param: float) -> None:
        self.lambda_threshold = cost_param
