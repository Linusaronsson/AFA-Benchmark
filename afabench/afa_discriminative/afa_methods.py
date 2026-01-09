import math
from copy import deepcopy
from pathlib import Path
from typing import Self, override

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchrl.modules import MLP

from afabench.afa_discriminative.models import (
    ConvNet,
    Predictor,
    ResNet18Backbone,
    resnet18,
)
from afabench.afa_discriminative.utils import (
    ConcreteSelector,
    MaskLayer,
    MaskLayer2d,
    get_entropy,
    ind_to_onehot,
    make_onehot,
    restore_parameters,
    patch_soft_to_feature_soft,
)
from afabench.common.custom_types import (
    AFAInitializer,
    AFAUnmasker,
    AFAMethod,
    AFASelection,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
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

    # def _patch_soft_to_feature_soft(
    #     self, soft_patch: torch.Tensor, x: torch.Tensor
    # ):
    #     """
    #     Convert soft patch mask (B, mask_size) into soft feature mask in x space.

    #     """
    #     if len(x.shape) == 4:
    #         B, C, H, W = x.shape
    #         mask_size = soft_patch.shape[1]
    #         mask_width = int(mask_size ** 0.5)
    #         m = soft_patch.view(B, 1, mask_width, mask_width)
    #         patch_size_h = H // mask_width
    #         patch_size_w = W // mask_width
    #         m = F.interpolate(m, scale_factor=(patch_size_h, patch_size_w), mode="nearest")
    #         return m.expand(B, C, H, W)

    #     assert len(x.shape) == 2
    #     B, D = x.shape
    #     mask_size = soft_patch.shape[1]

    #     if D == mask_size:
    #         return soft_patch

    #     # In case we use patch mask for tabular data
    #     assert D % mask_size == 0
    #     patch_len = D // mask_size
    #     return soft_patch.repeat_interleave(patch_len, dim=1)

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
            x, y = next(iter(val_loader))
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        if feature_costs is None:
            feature_costs = torch.ones(mask_size, device=device)
        log_cost = torch.log(feature_costs)
        # feature_shape = torch.Size([mask_size])
        # Pixel-evel feature masks
        x0, _ = next(iter(val_loader))
        x0 = x0.to(device)
        # TODO: we currently assume feature shape=data shape -> pixel mask for image data
        feature_shape = torch.Size(list(x0.shape[1:]))

        n_selections = unmasker.get_n_selections(feature_shape)
        msg = (
            f"Expected patch-level selection space to equal mask_size. "
            f"Got unmasker selection size={n_selections}, "
            f"mask_size={mask_size}."
        )
        assert n_selections == mask_size, msg

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

            for epoch in range(nepochs):
                # Switch models to training mode.
                selector.train()
                predictor.train()
                epoch_train_loss = 0.0
                for x_batch, y_batch in train_loader:
                    # Move to device.
                    x = x_batch.to(device)
                    y = y_batch.to(device)

                    # TODO: currently allow double selection
                    m_sel = torch.zeros(
                        len(x), mask_size, dtype=x.dtype, device=device
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
                        soft_patch = selector_layer(logits_cost, temp)
                        soft_feat = patch_soft_to_feature_soft(soft_patch, x)
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
                        # One-based indexing
                        afa_selection = sel_idx.to(torch.long) + 1
                        m_sel = torch.max(
                            m_sel,
                            make_onehot(dist),
                        )
                        # TODO: need to check if this work for image data
                        # TODO: also need to check the feature_shape
                        m_feat = unmasker.unmask(
                            masked_features=x_masked,
                            feature_mask=m_feat,
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
                        y = y_batch.to(device)

                        m_sel = torch.zeros(
                            len(x), mask_size, dtype=x.dtype, device=device
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
                                soft_patch = selector_layer(
                                    logits_cost, temp, deterministic=True
                                )
                            else:
                                soft_patch = selector_layer(logits_cost, temp)
                            soft_feat = patch_soft_to_feature_soft(soft_patch, x)
                            m_soft_feat = torch.maximum(m_feat, soft_feat)
                            m_sel = torch.max(m_sel, make_onehot(soft_patch))
                            sel_idx = torch.argmax(soft_patch, dim=1, keepdim=True)
                            afa_selection = sel_idx.to(torch.long) + 1
                            m_feat = unmasker.unmask(
                                masked_features=x_masked,
                                feature_mask=m_feat,
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

    @override
    def forward(
        self,
        x: torch.Tensor,
        max_features: int,
        argmax: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Make predictions using selected features."""
        # Setup.
        selector = self.selector
        predictor = self.predictor
        mask_layer = self.mask_layer
        selector_layer = self.selector_layer
        device = next(predictor.parameters()).device

        # Determine mask size.
        if mask_layer.mask_size is not None:
            mask_size = int(self.mask_layer.mask_size)  # pyright: ignore[reportArgumentType]
        else:
            # Must be tabular (1d data).
            assert len(x.shape) == 2
            mask_size = x.shape[1]
        m = torch.zeros(len(x), mask_size, device=device)

        for _ in range(max_features):
            # Evaluate selector model.
            x_masked = mask_layer(x, m)
            logits = selector(x_masked).flatten(1)

            # Update selections, ensure no repeats.
            logits = logits - 1e6 * m
            if argmax:
                m = torch.max(m, make_onehot(logits))
            else:
                m = torch.max(m, make_onehot(selector_layer(logits, 1e-6)))

        # Make predictions.
        x_masked = mask_layer(x, m)
        pred = predictor(x_masked)
        return pred, x_masked, m


class Covert2023AFAMethod(AFAMethod):
    def __init__(
        self,
        selector: nn.Module,
        predictor: nn.Module,
        device: torch.device,
        lambda_threshold: float | None = None,
        feature_costs: torch.Tensor | None = None,
        modality: str | None = "tabular",
        n_patches: int | None = None,
        d_in: int | None = None,
        d_out: int | None = None,
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
        self._feature_costs: torch.Tensor | None = feature_costs
        self.modality: str | None = modality
        # for image selection
        self.n_patches: int | None = n_patches
        self.d_in: int | None = d_in
        self.d_out: int | None = d_out
        self.image_size: int | None = None
        self.patch_size: int | None = None
        self.mask_width: int | None = None

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
    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFASelection:
        if self.modality == "tabular":
            x_masked = torch.cat([masked_features, feature_mask], dim=1)
            logits = self.selector(x_masked).flatten(1)
            logits = logits - 1e6 * feature_mask
        else:
            logits = self.selector(masked_features)
            assert logits.dim() == 2, (
                f"Selector must return [B, N], got {logits.shape}"
            )
            patch_mask = self._flat_mask_to_patch_mask(feature_mask).float()
            logits = logits - 1e6 * patch_mask

        if self._feature_costs is not None:
            costs = self._feature_costs.to(self._device)
            costs = torch.clamp(costs, min=1e-12)
            scores = logits / costs.unsqueeze(0)
        else:
            scores = logits
        best_scores, best_idx = scores.max(dim=1)
        lam = self.lambda_threshold
        stop_mask = best_scores < lam
        # all masked
        stop_mask = stop_mask | (best_scores < -1e5)

        selections = (best_idx + 1).to(dtype=torch.long).unsqueeze(-1)
        stop_mask = stop_mask.unsqueeze(-1)
        # 0 = stop
        selections = selections.masked_fill(stop_mask, 0)
        return selections
        # next_feature_idx = logits.argmax(dim=1)
        # return next_feature_idx

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(path / "model.pt", map_location=device)
        arch = checkpoint["architecture"]
        # tabular
        if "predictor_hidden_layers" in arch:
            d_in = arch["d_in"]
            d_out = arch["d_out"]
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
                out_features=d_in,
                num_cells=selector_hidden_layers,
                activation_class=nn.ReLU,
                dropout=dropout,
            )

            model = cls(selector, predictor, device)
            model.selector.load_state_dict(checkpoint["selector_state_dict"])
            model.predictor.load_state_dict(checkpoint["predictor_state_dict"])
            model.selector.eval()
            model.predictor.eval()
            return model.to(device)

        backbone = arch["backbone"]
        if backbone == "resnet18":
            d_out = arch["d_out"]
            base = resnet18(pretrained=False)
            backbone_net, expansion = ResNet18Backbone(base)
            predictor = Predictor(backbone_net, expansion, d_out)
            selector = ConvNet(backbone_net, 1, 0.5)

            model = cls(
                selector=selector,
                predictor=predictor,
                device=device,
                modality="image",
                n_patches=int(arch["mask_width"]) ** 2,
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
                "d_in": self.d_in,
                "d_out": self.d_out,
                "selector_hidden_layers": [128, 128],
                "predictor_hidden_layers": [128, 128],
                "dropout": 0.3,
                "model_type": "tabular",
            }
        else:
            # TODO: pass self.predictor.fc.out_features as d_out here
            arch = {
                "backbone": "resnet18",
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

        if feature_costs is None:
            feature_costs = torch.ones(mask_size).to(device)
        elif isinstance(feature_costs, np.ndarray):
            feature_costs = torch.tensor(feature_costs).to(device)
        # feature_shape = torch.Size([mask_size])
        x0, _ = next(iter(val_loader))
        x0 = x0.to(device)
        feature_shape = torch.Size(list(x0.shape[1:]))

        n_selections = unmasker.get_n_selections(feature_shape)
        msg = (
            f"Expected patch-level selection space to equal mask_size. "
            f"Got unmasker selection size={n_selections}, "
            f"mask_size={mask_size}."
        )
        assert n_selections == mask_size, msg

        # if initializer is not None:
        #     x0, y0 = next(iter(train_loader))
        #     x0 = x0.to(device)
        #     y0 = y0.to(device)
        #     init_mask_bool0 = initializer.initialize(
        #         features=x0,
        #         label=y0,
        #         feature_shape=feature_shape,
        #     ).to(device)
        #     num_initial = int(init_mask_bool0[0].sum().item())
        # else:
        #     num_initial = 0
        # effective_max_features = max(0, max_features - num_initial)

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

        for epoch in range(nepochs):
            # Switch models to training mode.
            value_network.train()
            predictor.train()
            value_losses = []
            pred_losses = []
            total_loss = 0

            for x_batch, y_batch in train_loader:
                # Move to device.
                x = x_batch.to(device)
                y = y_batch.to(device)

                m_sel = torch.zeros(
                    len(x), mask_size, dtype=x.dtype, device=device
                )
                init_mask_bool = initializer.initialize(
                    features=x,
                    label=y,
                    feature_shape=feature_shape,
                ).to(device)
                m_feat = init_mask_bool.to(dtype=x.dtype)

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

                    best = torch.argmax(pred_cmi / feature_costs, dim=1)
                    rng = np.random.default_rng()
                    random = torch.tensor(
                        rng.choice(mask_size, size=len(x)),
                        device=x.device,
                    )
                    exploit = (torch.rand(len(x), device=x.device) > eps).int()
                    actions = exploit * best + (1 - exploit) * random
                    # TODO: need to verify if this work for unmasking using image patch index
                    afa_selection = actions.to(torch.long) + 1
                    afa_selection = afa_selection.unsqueeze(1)
                    m_sel = torch.max(m_sel, ind_to_onehot(actions, mask_size))

                    # Predictor loss.
                    m_feat = unmasker.unmask(
                        masked_features=x_masked,
                        feature_mask=m_feat,
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
                    value_network_loss = nn.functional.mse_loss(
                        pred_cmi[torch.arange(len(x)), actions], delta
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

            # Calculate validation loss.
            value_network.eval()
            predictor.eval()
            val_preds = [[] for _ in range(max_features + 1)]
            val_targets = []

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    # Move to device.
                    x = x_batch.to(device)
                    y = y_batch.to(device)
                    m_sel = torch.zeros(
                        len(x), mask_size, dtype=x.dtype, device=device
                    )

                    # Setup.
                    init_mask_bool = initializer.initialize(
                        features=x,
                        label=y,
                        feature_shape=feature_shape,
                    ).to(device)
                    m_feat = init_mask_bool.to(dtype=x.dtype)
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
                            pred_cmi / feature_costs, dim=1
                        )
                        m_sel = torch.max(
                            m_sel, ind_to_onehot(best_feature_index, mask_size)
                        )
                        afa_selection = best_feature_index.to(torch.long) + 1
                        afa_selection = afa_selection.unsqueeze(1)
                        m_feat = unmasker.unmask(
                            masked_features=x_masked,
                            feature_mask=m_feat,
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

            # Print progress.
            if verbose:
                print(f"{'-' * 8}Epoch {epoch + 1}{'-' * 8}")
                print(f"Loss Val/Mean = {val_loss_mean}")
                print(f"Perf Val/Mean = {val_perf_mean}")
                print(f"Loss Val/Final = {val_loss_final}")
                print(f"Perf Val/Final = {val_perf_final}")
                print(f"Eps Value = {eps}\n")

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


class Gadgil2023AFAMethod(AFAMethod):
    def __init__(
        self,
        value_network: nn.Module,
        predictor: nn.Module,
        device: torch.device,
        lambda_threshold: float | None = None,
        feature_costs: torch.Tensor | None = None,
        modality: str | None = "tabular",
        n_patches: int | None = None,
        d_in: int | None = None,
        d_out: int | None = None,
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
        self._feature_costs: torch.Tensor | None = feature_costs
        self.modality: str | None = modality
        self.n_patches: int | None = n_patches
        self.d_in: int | None = d_in
        self.d_out: int | None = d_out
        self.image_size: int | None = None
        self.patch_size: int | None = None
        self.mask_width: int | None = None

    def _flat_mask_to_patch_mask(
        self, feature_mask: torch.Tensor
    ) -> AFASelection:
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
    def select(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFASelection:
        if self.modality == "tabular":
            x_masked = torch.cat([masked_features, feature_mask], dim=1)
            pred = self.predict(masked_features, feature_mask)
            entropy = get_entropy(pred).unsqueeze(1)
            pred_cmi = self.value_network(x_masked).sigmoid() * entropy
            pred_cmi -= 1e6 * feature_mask
        else:
            pred = self.predictor(masked_features)
            entropy = get_entropy(pred).unsqueeze(1)
            pred_cmi = self.value_network(masked_features).sigmoid() * entropy
            patch_mask = self._flat_mask_to_patch_mask(feature_mask).float()
            pred_cmi = pred_cmi - 1e6 * patch_mask

        if self._feature_costs is not None:
            costs = self._feature_costs.to(self._device)
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
        # next_feature_idx = torch.argmax(pred_cmi, dim=1)
        # return next_feature_idx

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(path / "model.pt", map_location=device)
        arch = checkpoint["architecture"]
        if "predictor_hidden_layers" in arch:
            d_in = arch["d_in"]
            d_out = arch["d_out"]
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
                out_features=d_in,
                num_cells=value_network_hidden_layers,
                activation_class=nn.ReLU,
                dropout=dropout,
            )
            # Tie weights
            # value_network.hidden[0] = predictor.hidden[0]
            # value_network.hidden[1] = predictor.hidden[1]
            pred_linears = [m for m in predictor if isinstance(m, nn.Linear)]
            value_linears = [
                m for m in value_network if isinstance(m, nn.Linear)
            ]

            assert len(value_network_hidden_layers) == len(
                predictor_hidden_layers
            )
            for i in range(len(value_network_hidden_layers)):
                value_linears[i].weight = pred_linears[i].weight
                value_linears[i].bias = pred_linears[i].bias

            model = cls(value_network, predictor, device)
            model.value_network.load_state_dict(
                checkpoint["value_network_state_dict"]
            )
            model.predictor.load_state_dict(checkpoint["predictor_state_dict"])
            model.value_network.eval()
            model.predictor.eval()
            return model.to(device)

        backbone = arch["backbone"]
        if backbone == "resnet18":
            d_out = arch["d_out"]
            base = resnet18(pretrained=False)
            backbone_net, expansion = ResNet18Backbone(base)
            predictor = Predictor(backbone_net, expansion, d_out)
            value_network = ConvNet(backbone_net, 1, 0.5)

            model = cls(
                value_network=value_network,
                predictor=predictor,
                device=device,
                modality="image",
                n_patches=int(arch["mask_width"]) ** 2,
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
                "d_in": self.d_in,
                "d_out": self.d_out,
                "value_network_hidden_layers": [128, 128],
                "predictor_hidden_layers": [128, 128],
                "dropout": 0.3,
                "model_type": "tabular",
            }
        else:
            arch = {
                "backbone": "resnet18",
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
