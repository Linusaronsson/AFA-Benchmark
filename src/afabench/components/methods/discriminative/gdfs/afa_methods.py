"""GDFS, greedy dynamic feature selection (Covert et al., ICML 2023)."""

from copy import deepcopy
from typing import ClassVar, override

import numpy as np
import torch
from torch import Tensor, nn

from afabench.components.methods.discriminative.common.afa_methods import (
    NetworkPairAFAMethod,
)
from afabench.components.methods.discriminative.common.trainer import (
    BatchEpisode,
    EpochValidation,
    TrainingContext,
    TrainingObjective,
    apply_model_mask,
)
from afabench.components.methods.discriminative.common.utils import (
    ConcreteSelector,
    get_entropy,
    make_onehot,
    patch_soft_to_feature_soft,
    restore_parameters,
    selection_soft_to_feature_soft,
)
from afabench.components.unmaskers.cube_nm_unmasker import CubeNMUnmasker
from afabench.core.types import (
    AFAAction,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


class GDFSAFAMethod(NetworkPairAFAMethod):
    """
    Acts on the selector's logits, and stops when the prediction is certain.

    The selector was trained through a Concrete relaxation to pick the
    selection that most improves the prediction; at evaluation its logit,
    per unit of selection cost, ranks the selections. The stop test is the
    classifier's entropy against `lambda_threshold`.
    """

    policy_network_name: ClassVar[str] = "selector"

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        network_input = self._network_input(masked_features, feature_mask)
        with torch.no_grad():
            pred = self.classifier(network_input)
            stop_mask = get_entropy(pred) < self.lambda_threshold
        logits = self.policy_network(network_input).flatten(1)
        scores = self._exclude_performed_selections(
            logits, feature_mask, selection_mask
        )
        return self._actions(self._score_per_cost(scores), stop_mask)


class GDFSTrainingObjective(TrainingObjective):
    """
    GDFS's step: a Concrete relaxation of the selection, annealed over rounds.

    Each round trains at one temperature of `np.geomspace(start_temp,
    end_temp, temp_steps)` with a fresh optimizer. In a training episode
    the selector's logits, less the log selection costs, are sampled
    through the Concrete distribution; the classifier sees the soft mask
    and its loss is backpropagated per selection; the hard argmax
    selection advances the episode. Validation scores the soft and the
    hard (zero-temperature) prediction; the networks best at zero
    temperature across all rounds are kept at the end.
    """

    name: ClassVar[str] = "gdfs"

    def __init__(
        self,
        *,
        start_temp: float = 1.0,
        end_temp: float = 0.1,
        temp_steps: int = 5,
        argmax: bool = False,
    ) -> None:
        super().__init__()
        self.temperatures: list[float] = [
            float(temp)
            for temp in np.geomspace(start_temp, end_temp, temp_steps)
        ]
        self.argmax: bool = argmax
        self.selector_layer: ConcreteSelector = ConcreteSelector()
        self.temp: float = start_temp
        self._log_cost: Tensor | None = None
        self._best_val: Tensor | None = None
        self._best_zerotemp_selector: nn.Module | None = None
        self._best_zerotemp_classifier: nn.Module | None = None
        self._epoch_train_loss: float = 0.0
        self._n_train_batches: int = 0
        self._pred_list: list[Tensor] = []
        self._hard_pred_list: list[Tensor] = []
        self._label_list: list[Tensor] = []

    @property
    @override
    def n_rounds(self) -> int:
        return len(self.temperatures)

    @override
    def prepare(self, context: TrainingContext) -> None:
        super().prepare(context)
        # Not a probability, so exp(logits) / cost is logits - log(cost)
        self._log_cost = torch.log(context.selection_costs)
        self._best_val = None
        self._best_zerotemp_selector = None
        self._best_zerotemp_classifier = None

    @override
    def start_round(self, round_index: int) -> None:
        self.temp = self.temperatures[round_index]
        if self.context.settings.verbose:
            print(f"Starting training with temp = {self.temp:.4f}\n")

    @override
    def start_epoch(self) -> None:
        self._epoch_train_loss = 0.0
        self._n_train_batches = 0
        self._pred_list = []
        self._hard_pred_list = []
        self._label_list = []

    def _soft_feature_mask(self, soft: Tensor, features: Tensor) -> Tensor:
        """Spread a soft selection over the features its feature group holds."""
        context = self.context
        if features.dim() == 4:
            return patch_soft_to_feature_soft(soft, features)
        if isinstance(context.unmasker, CubeNMUnmasker):
            return selection_soft_to_feature_soft(
                soft,
                mask_size=context.mask_size,
                n_contexts=context.unmasker.n_contexts,
            )
        return soft

    @override
    def train_episode(self, episode: BatchEpisode) -> None:
        context = self.context
        settings = context.settings
        selector = context.policy_network
        classifier = context.classifier
        assert self._log_cost is not None
        x = episode.features
        y = episode.label
        m_feat = episode.feature_mask
        m_sel = episode.selection_mask

        for _ in range(settings.selections_per_episode):
            # Evaluate selector model.
            x_masked = apply_model_mask(context.mask_layer, x, m_feat)
            logits = selector(x_masked).flatten(1)
            logits_cost = logits - self._log_cost

            # Get selections.
            soft = self.selector_layer(logits_cost, self.temp)
            m_soft_feat = torch.maximum(
                m_feat, self._soft_feature_mask(soft, x)
            )

            # Evaluate classifier model.
            x_masked = apply_model_mask(context.mask_layer, x, m_soft_feat)
            pred = classifier(x_masked)

            # Calculate loss.
            loss = settings.loss_fn(pred, y)
            (loss / settings.selections_per_episode).backward()
            self._epoch_train_loss += loss.item()

            # Update mask, ensure no repeats.
            dist = self.selector_layer(logits_cost - 1e6 * m_sel, 1e-6)
            afa_selection = torch.argmax(dist, dim=1, keepdim=True).to(
                torch.long
            )
            m_sel = torch.max(m_sel, make_onehot(dist))
            m_feat = context.unmasker.unmask(
                masked_features=x_masked,
                feature_mask=m_feat.bool(),
                features=x,
                afa_selection=afa_selection,
                selection_mask=m_sel,
                feature_shape=context.feature_shape,
            ).to(dtype=x.dtype)
        self._n_train_batches += 1

    @override
    def validate_episode(self, episode: BatchEpisode) -> None:
        context = self.context
        selector = context.policy_network
        classifier = context.classifier
        assert self._log_cost is not None
        x = episode.features
        y = episode.label
        m_feat = episode.feature_mask
        m_sel = episode.selection_mask

        for _ in range(context.settings.selections_per_episode):
            # Evaluate selector model.
            x_masked = apply_model_mask(context.mask_layer, x, m_feat)
            logits = selector(x_masked).flatten(1)
            logits_cost = logits - self._log_cost - 1e6 * m_sel

            # Get selections, ensure no repeats.
            soft = self.selector_layer(
                logits_cost, self.temp, deterministic=self.argmax
            )
            m_soft_feat = torch.maximum(
                m_feat, self._soft_feature_mask(soft, x)
            )
            m_sel = torch.max(m_sel, make_onehot(soft))
            afa_selection = torch.argmax(soft, dim=1, keepdim=True).to(
                torch.long
            )
            m_feat = context.unmasker.unmask(
                masked_features=x_masked,
                feature_mask=m_feat.bool(),
                features=x,
                afa_selection=afa_selection,
                selection_mask=m_sel,
                feature_shape=context.feature_shape,
            ).to(dtype=x.dtype)

            # Evaluate classifier with soft sample.
            x_masked = apply_model_mask(context.mask_layer, x, m_soft_feat)
            pred = classifier(x_masked)

            # Evaluate classifier with hard sample.
            x_masked = apply_model_mask(context.mask_layer, x, m_feat)
            hard_pred = classifier(x_masked)

            self._pred_list.append(pred)
            self._hard_pred_list.append(hard_pred)
            self._label_list.append(y)

    @override
    def end_epoch(self) -> EpochValidation:
        context = self.context
        pred = torch.cat(self._pred_list, 0)
        hard_pred = torch.cat(self._hard_pred_list, 0)
        y = torch.cat(self._label_list, 0)
        val_loss = context.val_loss_fn(pred, y)
        val_hard_loss = context.val_loss_fn(hard_pred, y)
        avg_train = self._epoch_train_loss / (
            self._n_train_batches * context.settings.selections_per_episode
        )

        # Check if best model with zero temperature.
        if (
            self._best_val is None
            or (
                context.val_loss_mode == "min"
                and val_hard_loss < self._best_val
            )
            or (
                context.val_loss_mode == "max"
                and val_hard_loss > self._best_val
            )
        ):
            self._best_val = val_hard_loss
            self._best_zerotemp_selector = deepcopy(context.policy_network)
            self._best_zerotemp_classifier = deepcopy(context.classifier)

        return EpochValidation(
            score=val_loss,
            metrics={
                "temperature": self.temp,
                "train_loss": float(avg_train),
                "val_loss": float(val_loss.mean().item()),
                "val_hard_loss": float(val_hard_loss.mean().item()),
            },
        )

    @override
    def finish(self) -> None:
        # Copy parameters from best model with zero temperature.
        assert self._best_zerotemp_selector is not None
        assert self._best_zerotemp_classifier is not None
        restore_parameters(
            self.context.policy_network, self._best_zerotemp_selector
        )
        restore_parameters(
            self.context.classifier, self._best_zerotemp_classifier
        )


__all__ = ["GDFSAFAMethod", "GDFSTrainingObjective"]
