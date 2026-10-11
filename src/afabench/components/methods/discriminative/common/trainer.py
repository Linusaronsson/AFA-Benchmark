"""
The training loop GDFS and DIME share, with each paper's step kept apart.

GDFS (Covert et al., ICML 2023) and DIME (Gadgil, Covert and Lee, 2023)
come from one upstream code base, and DIME's trainer was built on GDFS's.
`NetworkPairTrainer.fit` is the outer loop they shared: the validation
metric protocol, Adam with ReduceLROnPlateau, the best-epoch copies and
early-stopping counter, per-epoch metric logging, and the episodes of a
batch that the Initializer starts and the Unmasker advances. What each
paper does inside an episode, and how it reacts to a plateau, is its
`TrainingObjective`: `GDFSTrainingObjective` and `DIMETrainingObjective`
in the methods' own packages.

The differences between this loop and the upstream implementations are
written up in `docs/explanation/gdfs_dime_shared_code.md`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from jaxtyping import Float, Integer
from torch import Tensor, nn, optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from afabench.components.methods.discriminative.common.utils import (
    MaskLayer,
    MaskLayer2d,
    restore_parameters,
)
from afabench.core.types import AFAInitializer, AFAUnmasker

type ValLossMode = Literal["min", "max"]
type MetricLogger = Callable[[dict[str, float]], None]
type BatchFeatures = Float[Tensor, "batch *feature_shape"]  # noqa: F821
type ClassIndices = Integer[Tensor, "batch"]  # noqa: F821
type FloatMask = Float[Tensor, "batch *mask_shape"]  # noqa: F821
type SelectionCosts = Float[Tensor, "n_selections"]  # noqa: F821
type ValidationScore = Float[Tensor, ""]


@dataclass(frozen=True, kw_only=True)
class TrainerSettings:
    """
    The hyperparameters of the shared loop.

    `val_loss_fn` scores a validation epoch for the scheduler and the
    best-epoch check, in the direction `val_loss_mode` says; without one,
    the epoch is scored by `loss_fn`, minimised. `early_stopping_epochs`
    is the number of epochs without a new best after which a round stops
    (GDFS) or the exploration rate decays (DIME); it defaults to
    `patience + 1`, one more than the scheduler waits.
    """

    lr: float
    nepochs: int
    selections_per_episode: int
    """How many selections a training episode performs: the hard budget under unit costs."""
    loss_fn: nn.Module
    min_lr: float
    val_loss_fn: nn.Module | None = None
    val_loss_mode: ValLossMode | None = None
    factor: float = 0.2
    patience: int = 2
    early_stopping_epochs: int | None = None
    verbose: bool = True


@dataclass(frozen=True)
class BatchEpisode:
    """A batch of episodes at their initial feature mask, on the training device."""

    features: BatchFeatures
    label: ClassIndices
    feature_mask: FloatMask
    selection_mask: FloatMask


@dataclass(frozen=True)
class TrainingContext:
    """What `fit` resolved once before the first epoch, for the objective."""

    policy_network: nn.Module
    predictor: nn.Module
    mask_layer: MaskLayer | MaskLayer2d
    unmasker: AFAUnmasker
    device: torch.device
    settings: TrainerSettings
    val_loss_fn: nn.Module
    val_loss_mode: ValLossMode
    selection_costs: SelectionCosts
    n_selections: int
    mask_size: int
    feature_shape: torch.Size


@dataclass(frozen=True)
class EpochValidation:
    score: ValidationScore
    """What the scheduler steps on and the best-epoch check compares."""
    metrics: dict[str, float]


class TrainingObjective(ABC):
    """
    One paper's part of the loop: its episodes, and its reaction to a plateau.

    `fit` calls `prepare` once, then per round `start_round`, per epoch
    `start_epoch`, `train_episode` for each training batch with gradients
    zeroed before and the optimizer stepped after, `validate_episode` for
    each validation batch under `no_grad`, and `end_epoch` for the epoch's
    score and metrics. After `early_stopping_epochs` epochs without a new
    best it calls `on_plateau`, and the round ends when that answers True.
    `finish` runs once after the last round.
    """

    name: ClassVar[str]
    """The metric prefix, the pipeline's name for the method."""

    def __init__(self) -> None:
        self._context: TrainingContext | None = None

    @property
    def context(self) -> TrainingContext:
        if self._context is None:
            msg = f"{type(self).__name__} is used before fit prepared it"
            raise RuntimeError(msg)
        return self._context

    @property
    def n_rounds(self) -> int:
        """How many times the optimizer and scheduler are rebuilt and the epochs rerun."""
        return 1

    def prepare(self, context: TrainingContext) -> None:
        self._context = context

    def start_round(self, round_index: int) -> None:
        del round_index

    @abstractmethod
    def start_epoch(self) -> None: ...

    @abstractmethod
    def train_episode(self, episode: BatchEpisode) -> None: ...

    @abstractmethod
    def validate_episode(self, episode: BatchEpisode) -> None: ...

    @abstractmethod
    def end_epoch(self) -> EpochValidation: ...

    def on_plateau(self, optimizer: Optimizer) -> bool:
        """React to `early_stopping_epochs` epochs without a new best; answer whether the round stops."""
        del optimizer
        return True

    def finish(self) -> None:  # noqa: B027
        """Run once after the last round; a hook with nothing to do by default."""


def apply_model_mask(
    mask_layer: MaskLayer | MaskLayer2d,
    features: Tensor,
    feature_mask: Tensor,
) -> Tensor:
    """Mask `features` for the networks: a 2d mask layer gets pixel masks applied directly."""
    if isinstance(mask_layer, MaskLayer2d) and features.dim() == 4:
        return features * feature_mask
    return mask_layer(features, feature_mask)


def _to_class_indices(y: Tensor) -> ClassIndices:
    if y.ndim >= 2:
        return y.argmax(dim=-1).long()
    return y.long()


class NetworkPairTrainer:
    """Trains a policy network and a predictor with a `TrainingObjective`."""

    def __init__(
        self,
        *,
        policy_network: nn.Module,
        predictor: nn.Module,
        mask_layer: MaskLayer | MaskLayer2d,
        initializer: AFAInitializer,
        unmasker: AFAUnmasker,
        objective: TrainingObjective,
    ) -> None:
        self.policy_network: nn.Module = policy_network
        self.predictor: nn.Module = predictor
        self.mask_layer: MaskLayer | MaskLayer2d = mask_layer
        self.initializer: AFAInitializer = initializer
        self.unmasker: AFAUnmasker = unmasker
        self.objective: TrainingObjective = objective

    def fit(
        self,
        train_loader: DataLoader[tuple[Tensor, ...]],
        val_loader: DataLoader[tuple[Tensor, ...]],
        settings: TrainerSettings,
        *,
        feature_costs: Tensor | None = None,
        metric_logger: MetricLogger | None = None,
    ) -> None:
        """Train in place on the predictor's device; `feature_costs` default to unit costs."""
        context = self._prepare(settings, val_loader, feature_costs)
        self.objective.prepare(context)
        total_epochs = 0
        for round_index in range(self.objective.n_rounds):
            self.objective.start_round(round_index)
            total_epochs += self._fit_round(
                context, train_loader, val_loader, total_epochs, metric_logger
            )
        self.objective.finish()

    def _fit_round(
        self,
        context: TrainingContext,
        train_loader: DataLoader[tuple[Tensor, ...]],
        val_loader: DataLoader[tuple[Tensor, ...]],
        epochs_before: int,
        metric_logger: MetricLogger | None,
    ) -> int:
        """Run one round's epochs with a fresh optimizer; answer how many ran."""
        settings = context.settings
        policy_network = self.policy_network
        predictor = self.predictor
        objective = self.objective
        early_stopping_epochs = (
            settings.patience + 1
            if settings.early_stopping_epochs is None
            else settings.early_stopping_epochs
        )
        # Listing both networks' parameters, as the port always has, lists
        # a parameter DIME ties between them twice, so Adam steps it twice
        # per update; upstream deduplicates. Kept until #116 is decided
        # (docs/explanation/gdfs_dime_shared_code.md). The order is
        # immaterial: Adam updates each parameter from its own state
        opt = optim.Adam(
            list(policy_network.parameters()) + list(predictor.parameters()),
            lr=settings.lr,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=context.val_loss_mode,
            factor=settings.factor,
            patience=settings.patience,
            min_lr=settings.min_lr,
        )
        best_policy_network = deepcopy(policy_network)
        best_predictor = deepcopy(predictor)
        num_bad_epochs = 0
        epoch = 0

        for epoch in range(settings.nepochs):
            policy_network.train()
            predictor.train()
            objective.start_epoch()
            for x_batch, y_batch in train_loader:
                episode = self._start_episode(context, x_batch, y_batch)
                opt.zero_grad(set_to_none=True)
                objective.train_episode(episode)
                opt.step()

            policy_network.eval()
            predictor.eval()
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    objective.validate_episode(
                        self._start_episode(context, x_batch, y_batch)
                    )
                validation = objective.end_epoch()
            self._report_epoch(validation, epoch, epochs_before, metric_logger)

            scheduler.step(validation.score)
            if validation.score == scheduler.best:
                best_policy_network = deepcopy(policy_network)
                best_predictor = deepcopy(predictor)
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            if num_bad_epochs > early_stopping_epochs:
                num_bad_epochs = 0
                if objective.on_plateau(opt):
                    break

        if settings.verbose:
            print(f"Stopping the round at epoch {epoch + 1}\n")
        restore_parameters(policy_network, best_policy_network)
        restore_parameters(predictor, best_predictor)
        return epoch + 1

    def _report_epoch(
        self,
        validation: EpochValidation,
        epoch: int,
        epochs_before: int,
        metric_logger: MetricLogger | None,
    ) -> None:
        epochs_so_far = epoch + 1 + epochs_before
        if self.objective.context.settings.verbose:
            print(
                f"{'-' * 8}Epoch {epoch + 1} ({epochs_so_far} total){'-' * 8}"
            )
            for name, value in validation.metrics.items():
                print(f"{name} = {value:.4f}")
            print()
        if metric_logger is not None:
            metric_logger(
                {
                    f"{self.objective.name}/epoch": float(epochs_so_far),
                    **{
                        f"{self.objective.name}/{name}": value
                        for name, value in validation.metrics.items()
                    },
                }
            )

    def _prepare(
        self,
        settings: TrainerSettings,
        val_loader: DataLoader[tuple[Tensor, ...]],
        feature_costs: Tensor | None,
    ) -> TrainingContext:
        val_loss_fn = settings.val_loss_fn
        val_loss_mode = settings.val_loss_mode
        if val_loss_fn is None:
            val_loss_fn = settings.loss_fn
            val_loss_mode = "min"
        elif val_loss_mode is None:
            msg = (
                "must specify val_loss_mode (min or max) when val_loss_fn "
                "is specified"
            )
            raise ValueError(msg)
        device = next(self.predictor.parameters()).device
        val_loss_fn.to(device)

        x0, _ = next(iter(val_loader))
        feature_shape = torch.Size(list(x0.shape[1:]))
        mask_size = (
            feature_shape.numel()
            if self.mask_layer.mask_size is None
            else int(self.mask_layer.mask_size)
        )
        if feature_costs is None:
            feature_costs = torch.ones(
                feature_shape if len(feature_shape) == 3 else mask_size,
                device="cpu",
            )
        return TrainingContext(
            policy_network=self.policy_network,
            predictor=self.predictor,
            mask_layer=self.mask_layer,
            unmasker=self.unmasker,
            device=device,
            settings=settings,
            val_loss_fn=val_loss_fn,
            val_loss_mode=val_loss_mode,
            selection_costs=self.unmasker.get_selection_costs(
                feature_costs
            ).to(device),
            n_selections=self.unmasker.get_n_selections(feature_shape),
            mask_size=mask_size,
            feature_shape=feature_shape,
        )

    def _start_episode(
        self, context: TrainingContext, x_batch: Tensor, y_batch: Tensor
    ) -> BatchEpisode:
        features = x_batch.to(context.device)
        label = _to_class_indices(y_batch).to(context.device)
        selection_mask = torch.zeros(
            len(features),
            context.n_selections,
            dtype=features.dtype,
            device=context.device,
        )
        feature_mask = self.initializer.initialize(
            features=features,
            label=label,
            feature_shape=context.feature_shape,
        ).to(context.device)
        return BatchEpisode(
            features=features,
            label=label,
            feature_mask=feature_mask.to(dtype=features.dtype),
            selection_mask=selection_mask,
        )
