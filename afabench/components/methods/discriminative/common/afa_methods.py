import math
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any, Self, override

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchrl.modules import MLP

from afabench.components.methods.discriminative.common.models import (
    ConvNet,
    Predictor,
    ResNet18Backbone,
    resnet18,
    resnet50,
)
from afabench.components.methods.discriminative.common.utils import (
    ConcreteSelector,
    MaskLayer,
    MaskLayer2d,
    get_entropy,
    ind_to_onehot,
    make_onehot,
    patch_soft_to_feature_soft,
    restore_parameters,
    selection_soft_to_feature_soft,
)
from afabench.components.unmaskers import CubeNMUnmasker
from afabench.core.types import (
    AFAAction,
    AFAInitializer,
    AFAMethod,
    AFAUnmasker,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


def _apply_model_mask(
    mask_layer: MaskLayer | MaskLayer2d,
    features: torch.Tensor,
    feature_mask: torch.Tensor,
) -> torch.Tensor:
    if isinstance(mask_layer, MaskLayer2d) and features.dim() == 4:
        return features * feature_mask
    return mask_layer(features, feature_mask)


def _append_flat_mask(
    masked_features: torch.Tensor, feature_mask: torch.Tensor
) -> torch.Tensor:
    if masked_features.dim() > 2:
        masked_features = masked_features.flatten(start_dim=1)
        feature_mask = feature_mask.flatten(start_dim=1)
    return torch.cat([masked_features, feature_mask], dim=1)


def _unpack_training_batch(
    batch: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    unmasker: AFAUnmasker,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return values, labels, factual support, and selectable support."""
    features, labels = batch[:2]
    features = features.to(device)
    labels = labels.to(device)
    if len(batch) == 4:
        source_availability = batch[2].to(device).bool()
        feature_availability = batch[3].to(device).bool()
    else:
        source_availability = torch.ones_like(features, dtype=torch.bool)
        feature_availability = source_availability
    selection_availability = (
        unmasker.feature_availability_to_selection_availability(
            feature_availability
        )
    )
    return features, labels, source_availability, selection_availability


def _initial_training_masks(
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    feature_shape: torch.Size,
    initializer: AFAInitializer,
    selection_availability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    initial_features = initializer.initialize(
        features=features,
        label=labels,
        feature_shape=feature_shape,
    ).to(features.device)
    # Materialized restricted views contain zeros outside support. Their
    # unavailable actions begin as already exhausted in selection space.
    selection_mask = (~selection_availability).to(dtype=features.dtype)
    return initial_features.to(dtype=features.dtype), selection_mask


def _unmask_available_rows(
    *,
    unmasker: AFAUnmasker,
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
    features: torch.Tensor,
    selection: torch.Tensor,
    selection_mask: torch.Tensor,
    has_available: torch.Tensor,
    feature_shape: torch.Size,
) -> torch.Tensor:
    new_feature_mask = feature_mask.clone().bool()
    # One `nonzero` shared by the five gathers below, rather than five boolean
    # index expansions. The `has_available.any()` guard it replaces cost a
    # device sync to answer a question the empty-index case answers for free.
    rows = has_available.nonzero(as_tuple=True)[0]
    new_feature_mask[rows] = unmasker.unmask(
        masked_features=masked_features[rows],
        feature_mask=feature_mask[rows].bool(),
        features=features[rows],
        afa_selection=selection[rows],
        selection_mask=selection_mask[rows],
        feature_shape=feature_shape,
    )
    return new_feature_mask.to(dtype=feature_mask.dtype)


def _feature_marginal_selection_propensities(
    feature_availability: torch.Tensor,
    unmasker: AFAUnmasker,
) -> torch.Tensor:
    marginal = feature_availability.float().flatten(start_dim=1).mean(dim=0)
    if isinstance(unmasker, CubeNMUnmasker):
        return torch.cat(
            [
                marginal[: unmasker.n_contexts].prod().unsqueeze(0),
                marginal[unmasker.n_contexts :],
            ]
        )
    return marginal


def _training_selection_propensities(
    train_loader: DataLoader[Any],
    unmasker: AFAUnmasker,
    *,
    n_selections: int,
    device: torch.device,
) -> torch.Tensor:
    """Estimate feature-marginal support once from the full training view."""
    source_availability = getattr(
        train_loader.dataset,
        "source_availability",
        None,
    )
    if source_availability is None:
        return torch.ones(n_selections, device=device)
    return _feature_marginal_selection_propensities(
        source_availability.to(device),
        unmasker,
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

    def _to_class_indices(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim >= 2:
            return y.argmax(dim=-1).long()
        return y.long()

    def fit(  # noqa: PLR0915, PLR0912, C901
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        lr: float,
        nepochs: int,
        max_features: int,
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
        metric_logger: Callable[[dict[str, float]], None] | None = None,
        metric_prefix: str = "gdfs",
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
            x = next(iter(val_loader))[0]
            mask_size = x.shape[1:].numel()

        x0 = next(iter(val_loader))[0]
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
        selection_costs = unmasker.get_selection_costs(feature_costs).to(
            device
        )
        log_cost = torch.log(selection_costs)

        # For tracking best models with zero temperature.
        best_val = None
        best_zerotemp_selector = None
        best_zerotemp_predictor = None

        # Train separately with each temperature.
        total_epochs = 0
        for temp in np.geomspace(start_temp, end_temp, temp_steps):
            if verbose:
                print(f"Starting training with temp = {temp:.4f}\n")

            # Set up optimizer and lr scheduler.
            opt = optim.Adam(
                list(predictor.parameters()) + list(selector.parameters()),
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

            for epoch in range(nepochs):
                # Switch models to training mode.
                selector.train()
                predictor.train()
                # Accumulated on device and read back once per epoch. The
                # inner loop runs once per feature per batch, and a `.item()`
                # there is a full sync each time. float64 so the sum matches
                # the Python-float accumulation this replaces exactly, since
                # `avg_train` drives early stopping.
                epoch_train_loss = torch.zeros(
                    (), device=device, dtype=torch.float64
                )
                for batch in train_loader:
                    # Move to device.
                    (
                        x,
                        y_batch,
                        _source_availability,
                        selection_availability,
                    ) = _unpack_training_batch(
                        batch,
                        unmasker=unmasker,
                        device=device,
                    )
                    y = self._to_class_indices(y_batch).to(device)

                    m_feat, m_sel = _initial_training_masks(
                        features=x,
                        labels=y,
                        feature_shape=feature_shape,
                        initializer=initializer,
                        selection_availability=selection_availability,
                    )

                    selector.zero_grad()
                    predictor.zero_grad()

                    for _ in range(max_features):
                        # Evaluate selector model.
                        # x_masked = mask_layer(x, m_feat)
                        x_masked = _apply_model_mask(mask_layer, x, m_feat)
                        logits = selector(x_masked).flatten(1)
                        # since not a probability, do exp(logits)/cost <-> logits / log_cost
                        logits_cost = logits - log_cost - 1e6 * m_sel
                        has_available = (~m_sel.bool()).any(dim=1)

                        # Get selections.
                        # soft = selector_layer(logits, temp)
                        soft = selector_layer(logits_cost, temp)
                        soft *= has_available.unsqueeze(1)
                        if len(x.shape) == 4:
                            soft_feat = patch_soft_to_feature_soft(soft, x)
                        elif isinstance(unmasker, CubeNMUnmasker):
                            soft_feat = selection_soft_to_feature_soft(
                                soft,
                                mask_size=mask_size,
                                n_contexts=unmasker.n_contexts,
                            )
                        else:
                            soft_feat = soft
                        m_soft_feat = torch.maximum(m_feat, soft_feat)

                        # Evaluate predictor model.
                        x_masked = _apply_model_mask(
                            mask_layer, x, m_soft_feat
                        )
                        pred = predictor(x_masked)

                        # Calculate loss.
                        loss = loss_fn(pred, y)
                        (loss / max_features).backward()
                        epoch_train_loss += loss.detach().double()

                        # Update mask, ensure no repeats.
                        dist = selector_layer(logits_cost, 1e-6)
                        dist *= has_available.unsqueeze(1)
                        sel_idx = torch.argmax(dist, dim=1, keepdim=True)
                        # Zero-based indexing for unmaskers
                        afa_selection = sel_idx.to(torch.long)
                        m_sel = torch.max(
                            m_sel,
                            make_onehot(dist),
                        )
                        m_feat = _unmask_available_rows(
                            unmasker=unmasker,
                            masked_features=x_masked,
                            feature_mask=m_feat,
                            features=x,
                            selection=afa_selection,
                            selection_mask=m_sel,
                            has_available=has_available,
                            feature_shape=feature_shape,
                        ).to(dtype=x.dtype)

                    # Take gradient step.
                    opt.step()

                avg_train = epoch_train_loss.item() / (
                    len(train_loader) * max_features
                )

                # Calculate validation loss.
                selector.eval()
                predictor.eval()
                with torch.no_grad():
                    # For mean loss.
                    pred_list = []
                    hard_pred_list = []
                    label_list = []

                    for batch in val_loader:
                        # Move to device.
                        (
                            x,
                            y_batch,
                            _source_availability,
                            selection_availability,
                        ) = _unpack_training_batch(
                            batch,
                            unmasker=unmasker,
                            device=device,
                        )
                        y = self._to_class_indices(y_batch).to(device)

                        m_feat, m_sel = _initial_training_masks(
                            features=x,
                            labels=y,
                            feature_shape=feature_shape,
                            initializer=initializer,
                            selection_availability=selection_availability,
                        )

                        for _ in range(max_features):
                            # Evaluate selector model.
                            x_masked = _apply_model_mask(mask_layer, x, m_feat)
                            logits = selector(x_masked).flatten(1)
                            logits_cost = logits - log_cost
                            logits_cost = logits_cost - 1e6 * m_sel
                            has_available = (~m_sel.bool()).any(dim=1)

                            # Get selections, ensure no repeats.
                            # logits = logits - 1e6 * m
                            if argmax:
                                soft = selector_layer(
                                    logits_cost, temp, deterministic=True
                                )
                            else:
                                soft = selector_layer(logits_cost, temp)
                            soft *= has_available.unsqueeze(1)
                            if len(x.shape) == 4:
                                soft_feat = patch_soft_to_feature_soft(soft, x)
                            elif isinstance(unmasker, CubeNMUnmasker):
                                soft_feat = selection_soft_to_feature_soft(
                                    soft, mask_size, unmasker.n_contexts
                                )
                            else:
                                soft_feat = soft
                            m_soft_feat = torch.maximum(m_feat, soft_feat)
                            m_sel = torch.max(m_sel, make_onehot(soft))
                            sel_idx = torch.argmax(soft, dim=1, keepdim=True)
                            afa_selection = sel_idx.to(torch.long)
                            m_feat = _unmask_available_rows(
                                unmasker=unmasker,
                                masked_features=x_masked,
                                feature_mask=m_feat,
                                features=x,
                                selection=afa_selection,
                                selection_mask=m_sel,
                                has_available=has_available,
                                feature_shape=feature_shape,
                            ).to(dtype=x.dtype)

                            # Evaluate predictor with soft sample.
                            x_masked = _apply_model_mask(
                                mask_layer, x, m_soft_feat
                            )
                            pred = predictor(x_masked)

                            # Evaluate predictor with hard sample.
                            x_masked = _apply_model_mask(mask_layer, x, m_feat)
                            hard_pred = predictor(x_masked)

                            # Append predictions and labels.
                            pred_list.append(pred)
                            hard_pred_list.append(hard_pred)
                            label_list.append(y)

                    # Calculate mean loss.
                    pred = torch.cat(pred_list, 0)
                    hard_pred = torch.cat(hard_pred_list, 0)
                    y = torch.cat(label_list, 0)
                    val_loss = val_loss_fn(pred, y)
                    val_hard_loss = val_loss_fn(hard_pred, y)

                # Print progress.
                if verbose:
                    print(
                        f"{'-' * 8}Epoch {epoch + 1} ({
                            epoch + 1 + total_epochs
                        } total){'-' * 8}"
                    )
                    print(
                        f"Val loss = {val_loss:.4f}, Zero-temp loss = {val_hard_loss:.4f}\n"
                    )

                if metric_logger is not None:
                    metric_logger(
                        {
                            f"{metric_prefix}/epoch": float(
                                epoch + 1 + total_epochs
                            ),
                            f"{metric_prefix}/temperature": float(temp),
                            f"{metric_prefix}/train_loss": float(avg_train),
                            f"{metric_prefix}/val_loss": float(
                                val_loss.mean().item()
                            ),
                            f"{metric_prefix}/val_hard_loss": float(
                                val_hard_loss.mean().item()
                            ),
                        }
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
                print(f"Stopping temp = {temp:.4f} at epoch {epoch + 1}\n")
            total_epochs += epoch + 1

            # Copy parameters from best model.
            restore_parameters(selector, best_selector)
            restore_parameters(predictor, best_predictor)

        # Copy parameters from best model with zero temperature.
        assert best_zerotemp_selector is not None
        assert best_zerotemp_predictor is not None
        restore_parameters(selector, best_zerotemp_selector)
        restore_parameters(predictor, best_zerotemp_predictor)


class GDFSAFAMethod(AFAMethod):
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
        self.selector_hidden_layers: list[int] = selector_hidden_layers or [
            128,
            128,
        ]
        self.predictor_hidden_layers: list[int] = predictor_hidden_layers or [
            128,
            128,
        ]
        self.dropout: float = dropout
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
            x_masked = _append_flat_mask(masked_features, feature_mask)
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
                x_masked_pred = _append_flat_mask(
                    masked_features, feature_mask
                )
                pred = self.predictor(x_masked_pred)
            else:
                pred = self.predictor(masked_features)

            entropy = get_entropy(pred)
            stop_mask = entropy < self.lambda_threshold

        if self.modality == "tabular":
            x_masked = _append_flat_mask(masked_features, feature_mask)
            logits = self.selector(x_masked).flatten(1)
            # TODO: currently assume that if we use CubeNMUnmasker, then we
            # have a non-None selection mask
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
        checkpoint = torch.load(
            path / "model.pt", weights_only=False, map_location=device
        )
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
            "selection_costs": self._selection_costs.detach().cpu()
            if self._selection_costs is not None
            else None,
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
    ):
        super().__init__()

        # Save network modules.
        self.value_network: nn.Module = value_network
        self.predictor: nn.Module = predictor
        self.mask_layer: MaskLayer | MaskLayer2d = mask_layer
        self.initializer: AFAInitializer = initializer
        self.unmasker: AFAUnmasker = unmasker

    def _to_class_indices(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim >= 2:
            return y.argmax(dim=-1).long()
        return y.long()

    def fit(  # noqa: PLR0915, PLR0912, C901
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        lr: float,
        nepochs: int,
        max_features: int,
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
        metric_logger: Callable[[dict[str, float]], None] | None = None,
        metric_prefix: str = "cmi_estimator",
    ) -> None:
        if val_loss_fn is None:
            val_loss_fn = loss_fn
            val_loss_mode = "min"
        elif val_loss_mode is None:
            msg = "must specify val_loss_mode (min or max) when validation_loss_fn is specified"
            raise ValueError(msg)
        if early_stopping_epochs is None:
            early_stopping_epochs = patience + 1
        if ipw_mode not in {"none", "feature_marginal"}:
            msg = "ipw_mode must be one of {'none', 'feature_marginal'}."
            raise ValueError(msg)
        if not 0.0 < ipw_min_propensity <= 1.0:
            msg = "ipw_min_propensity must be in (0, 1]."
            raise ValueError(msg)
        if ipw_max_weight <= 0.0:
            msg = "ipw_max_weight must be positive."
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
            x = next(iter(val_loader))[0]
            mask_size = x.shape[1:].numel()

        x0 = next(iter(val_loader))[0]
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
        selection_costs = unmasker.get_selection_costs(feature_costs).to(
            device
        )
        selection_costs = torch.clamp(selection_costs, min=1e-12)

        n_selections = unmasker.get_n_selections(feature_shape)
        selection_propensities = _training_selection_propensities(
            train_loader,
            unmasker,
            n_selections=n_selections,
            device=device,
        )

        opt = optim.Adam(
            list(value_network.parameters()) + list(predictor.parameters()),
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

        for epoch in range(nepochs):
            # Switch models to training mode.
            value_network.train()
            predictor.train()
            value_losses = []
            pred_losses = []
            total_loss = 0

            for batch in train_loader:
                # Move to device.
                x, y_batch, _source_availability, selection_availability = (
                    _unpack_training_batch(
                        batch,
                        unmasker=unmasker,
                        device=device,
                    )
                )
                y = self._to_class_indices(y_batch).to(device)

                m_feat, m_sel = _initial_training_masks(
                    features=x,
                    labels=y,
                    feature_shape=feature_shape,
                    initializer=initializer,
                    selection_availability=selection_availability,
                )

                value_network.zero_grad()
                predictor.zero_grad()
                value_network_loss_total = 0
                pred_loss_total = 0

                # Predictor loss with initial features.
                x_masked = _apply_model_mask(mask_layer, x, m_feat)
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
                    x_masked = _apply_model_mask(mask_layer, x, m_feat)
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

                    available = ~m_sel.bool()
                    has_available = available.any(dim=1)
                    scores = (pred_cmi / selection_costs).masked_fill(
                        ~available,
                        -torch.inf,
                    )
                    best = torch.argmax(scores, dim=1)
                    random_scores = torch.rand_like(pred_cmi).masked_fill(
                        ~available,
                        -1.0,
                    )
                    random = torch.argmax(random_scores, dim=1)
                    exploit = (torch.rand(len(x), device=x.device) > eps).int()
                    actions = exploit * best + (1 - exploit) * random
                    afa_selection = actions.to(torch.long)
                    afa_selection = afa_selection.unsqueeze(1)
                    performed = ind_to_onehot(actions, n_selections)
                    performed *= has_available.unsqueeze(1)
                    m_sel = torch.max(
                        m_sel,
                        performed,
                    )

                    # Predictor loss.
                    m_feat = _unmask_available_rows(
                        unmasker=unmasker,
                        masked_features=x_masked,
                        feature_mask=m_feat,
                        features=x,
                        selection=afa_selection,
                        selection_mask=m_sel,
                        has_available=has_available,
                        feature_shape=feature_shape,
                    )
                    x_masked = _apply_model_mask(self.mask_layer, x, m_feat)
                    pred_with_next_feature = predictor(x_masked)
                    loss_with_next_feature = loss_fn(pred_with_next_feature, y)

                    # Value network loss.
                    delta = (
                        loss_without_next_feature
                        - loss_with_next_feature.detach()
                    )
                    squared_error = torch.square(
                        pred_cmi[
                            torch.arange(len(x), device=device),
                            actions,
                        ]
                        - delta
                    )
                    squared_error *= has_available
                    if ipw_mode == "feature_marginal":
                        weights = (
                            selection_propensities[actions]
                            .clamp_min(ipw_min_propensity)
                            .reciprocal()
                        )
                        weights = weights.clamp_max(ipw_max_weight)
                        if ipw_normalize_weights and has_available.any():
                            weights /= (
                                weights[has_available].mean().clamp_min(1e-12)
                            )
                        squared_error *= weights.detach()
                    value_network_loss = squared_error.sum() / (
                        has_available.sum().clamp_min(1)
                    )

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

            train_value_loss = torch.stack(value_losses).mean()
            train_pred_loss = torch.stack(pred_losses).mean()

            # Calculate validation loss.
            value_network.eval()
            predictor.eval()
            val_preds = [[] for _ in range(max_features + 1)]
            val_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    # Move to device.
                    (
                        x,
                        y_batch,
                        _source_availability,
                        selection_availability,
                    ) = _unpack_training_batch(
                        batch,
                        unmasker=unmasker,
                        device=device,
                    )
                    y = self._to_class_indices(y_batch).to(device)

                    # Setup.
                    m_feat, m_sel = _initial_training_masks(
                        features=x,
                        labels=y,
                        feature_shape=feature_shape,
                        initializer=initializer,
                        selection_availability=selection_availability,
                    )
                    x_masked = _apply_model_mask(self.mask_layer, x, m_feat)
                    pred = predictor(x_masked)
                    val_preds[0].append(pred)

                    for i in range(1, max_features + 1):
                        # Estimate CMI using value network.
                        x_masked = _apply_model_mask(mask_layer, x, m_feat)
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
                        has_available = (~m_sel.bool()).any(dim=1)
                        best_feature_index = torch.argmax(
                            pred_cmi / selection_costs, dim=1
                        )
                        performed = ind_to_onehot(
                            best_feature_index,
                            n_selections,
                        )
                        performed *= has_available.unsqueeze(1)
                        m_sel = torch.max(
                            m_sel,
                            performed,
                        )
                        afa_selection = best_feature_index.to(torch.long)
                        afa_selection = afa_selection.unsqueeze(1)
                        m_feat = _unmask_available_rows(
                            unmasker=unmasker,
                            masked_features=x_masked,
                            feature_mask=m_feat,
                            features=x,
                            selection=afa_selection,
                            selection_mask=m_sel,
                            has_available=has_available,
                            feature_shape=feature_shape,
                        )

                        # Make prediction.
                        x_masked = _apply_model_mask(
                            self.mask_layer, x, m_feat
                        )
                        pred = self.predictor(x_masked)
                        val_preds[i].append(pred)

                    val_targets.append(y)

                # Calculate mean loss.
                y_val = torch.cat(val_targets)
                preds_cat = [torch.cat(p) for p in val_preds]
                val_pred_losses = [loss_fn(p, y_val).mean() for p in preds_cat]
                val_scores = [val_loss_fn(p, y_val) for p in preds_cat]
                val_loss_mean = torch.stack(val_pred_losses).mean()
                val_perf_mean = torch.stack(val_scores).mean()
                val_loss_final = val_pred_losses[-1]
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

            # Print progress.
            if verbose:
                print(f"{'-' * 8}Epoch {epoch + 1}{'-' * 8}")
                print(f"Loss Val/Mean = {val_loss_mean}")
                print(f"Perf Val/Mean = {val_perf_mean}")
                print(f"Loss Val/Final = {val_loss_final}")
                print(f"Perf Val/Final = {val_perf_final}")
                print(f"Eps Value = {eps}\n")

            if metric_logger is not None:
                metric_logger(
                    {
                        f"{metric_prefix}/epoch": float(epoch + 1),
                        f"{metric_prefix}/train_value_loss": float(
                            train_value_loss.item()
                        ),
                        f"{metric_prefix}/train_predictor_loss": float(
                            train_pred_loss.item()
                        ),
                        f"{metric_prefix}/val_loss_mean": float(
                            val_loss_mean.mean().item()
                        ),
                        f"{metric_prefix}/val_perf_mean": float(
                            val_perf_mean.mean().item()
                        ),
                        f"{metric_prefix}/val_loss_final": float(
                            val_loss_final.mean().item()
                        ),
                        f"{metric_prefix}/val_perf_final": float(
                            val_perf_final.mean().item()
                        ),
                        f"{metric_prefix}/epsilon": float(eps),
                    }
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
                print(f"Decaying eps to {eps:.5f}, step = {num_epsilon_steps}")

                # Early stopping.
                if num_epsilon_steps >= eps_steps:
                    break

                # Reset optimizer learning rate. Could fully reset optimizer and scheduler, but this is simpler.
                for g in opt.param_groups:
                    g["lr"] = lr

        # Copy parameters from best model.
        restore_parameters(value_network, best_value_network)
        restore_parameters(predictor, best_predictor)


class DIMEAFAMethod(AFAMethod):
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
        self.value_network_hidden_layers: list[int] = (
            value_network_hidden_layers or [128, 128]
        )
        self.predictor_hidden_layers: list[int] = predictor_hidden_layers or [
            128,
            128,
        ]
        self.dropout: float = dropout
        self.modality: str | None = modality
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
            x_masked = _append_flat_mask(masked_features, feature_mask)
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
            x_masked = _append_flat_mask(masked_features, feature_mask)
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
        return selections

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(
            path / "model.pt", weights_only=False, map_location=device
        )
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
            "selection_costs": self._selection_costs.detach().cpu()
            if self._selection_costs is not None
            else None,
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
