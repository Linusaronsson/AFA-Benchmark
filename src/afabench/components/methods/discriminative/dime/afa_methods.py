"""DIME, discriminative mutual information estimation (Gadgil, Covert and Lee, 2023)."""

from typing import ClassVar, Literal, override

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

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
    get_entropy,
    ind_to_onehot,
)
from afabench.core.types import (
    AFAAction,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)


class DIMEAFAMethod(NetworkPairAFAMethod):
    """
    Acts on the value network's CMI estimates, and stops when none is worth its cost.

    The value network estimates each selection's conditional mutual
    information with the label, bounded by the predictor's entropy; at
    evaluation the estimate per unit of selection cost ranks the
    selections, and falls below `lambda_threshold` to stop.
    """

    policy_network_name: ClassVar[str] = "value_network"

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
        pred = self.predictor(network_input)
        entropy = get_entropy(pred).unsqueeze(1)
        pred_cmi = self.policy_network(network_input).sigmoid() * entropy
        scores = self._score_per_cost(
            self._exclude_performed_selections(
                pred_cmi, feature_mask, selection_mask
            )
        )
        stop_mask = scores.max(dim=1).values < self.lambda_threshold
        return self._actions(scores, stop_mask)


type CMIScaling = Literal["none", "positive", "bounded"]


class DIMETrainingObjective(TrainingObjective):
    """
    DIME's step: regress each selection's CMI on the loss it saves.

    In a training episode the value network estimates every selection's
    conditional mutual information with the label (bounded by the
    predictor's entropy under `cmi_scaling="bounded"`); the selection is
    the best estimate per unit of cost or, with probability `eps`, a
    random one. The value network is regressed on the predictor's loss
    drop from the selection, the predictor on its loss, with one backward
    pass over the episode. On a plateau `eps` decays by `eps_decay` and
    the learning rate is reset; after `eps_steps` decays training stops.
    """

    name: ClassVar[str] = "dime"

    def __init__(
        self,
        *,
        eps: float,
        eps_decay: float = 0.2,
        eps_steps: int = 1,
        cmi_scaling: CMIScaling = "bounded",
    ) -> None:
        super().__init__()
        self.initial_eps: float = eps
        self.eps: float = eps
        self.eps_decay: float = eps_decay
        self.eps_steps: int = eps_steps
        self.cmi_scaling: CMIScaling = cmi_scaling
        self._selection_costs: Tensor | None = None
        self._num_epsilon_steps: int = 0
        self._value_losses: list[Tensor] = []
        self._pred_losses: list[Tensor] = []
        self._val_preds: list[list[Tensor]] = []
        self._val_targets: list[Tensor] = []

    @override
    def prepare(self, context: TrainingContext) -> None:
        super().prepare(context)
        self._selection_costs = torch.clamp(context.selection_costs, min=1e-12)
        self.eps = self.initial_eps
        self._num_epsilon_steps = 0

    @override
    def start_epoch(self) -> None:
        self._value_losses = []
        self._pred_losses = []
        self._val_preds = [
            [] for _ in range(self.context.settings.selections_per_episode + 1)
        ]
        self._val_targets = []

    def _estimate_cmi(self, x_masked: Tensor, pred: Tensor) -> Tensor:
        value_network = self.context.policy_network
        if self.cmi_scaling == "bounded":
            entropy = get_entropy(pred).unsqueeze(1)
            return value_network(x_masked).sigmoid() * entropy
        if self.cmi_scaling == "positive":
            return torch.nn.functional.softplus(value_network(x_masked))
        return value_network(x_masked)

    @override
    def train_episode(self, episode: BatchEpisode) -> None:
        context = self.context
        settings = context.settings
        predictor = context.predictor
        loss_fn = settings.loss_fn
        assert self._selection_costs is not None
        x = episode.features
        y = episode.label
        m_feat = episode.feature_mask
        m_sel = episode.selection_mask
        value_network_loss_total = torch.zeros((), device=x.device)
        pred_loss_total = torch.zeros((), device=x.device)
        step_losses: list[Tensor] = []

        # Predictor loss with initial features.
        x_masked = apply_model_mask(context.mask_layer, x, m_feat)
        pred_without_next_feature = predictor(x_masked)
        loss_without_next_feature = loss_fn(pred_without_next_feature, y)
        pred_loss = loss_without_next_feature.mean()
        pred_loss_total += pred_loss.detach()

        step_losses.append(pred_loss)
        pred_without_next_feature = pred_without_next_feature.detach()
        loss_without_next_feature = loss_without_next_feature.detach()

        for _ in range(settings.selections_per_episode):
            # Estimate CMI using value network.
            x_masked = apply_model_mask(context.mask_layer, x, m_feat)
            pred_cmi = self._estimate_cmi(x_masked, pred_without_next_feature)

            best = torch.argmax(pred_cmi / self._selection_costs, dim=1)
            random = torch.randint(
                context.n_selections, size=(len(x),), device=x.device
            )
            exploit = (torch.rand(len(x), device=x.device) > self.eps).int()
            actions = exploit * best + (1 - exploit) * random
            afa_selection = actions.to(torch.long).unsqueeze(1)
            m_sel = torch.max(
                m_sel, ind_to_onehot(actions, context.n_selections)
            )

            # Predictor loss.
            m_feat = context.unmasker.unmask(
                masked_features=x_masked,
                feature_mask=m_feat.bool(),
                features=x,
                afa_selection=afa_selection,
                selection_mask=m_sel,
                feature_shape=context.feature_shape,
            )
            x_masked = apply_model_mask(context.mask_layer, x, m_feat)
            pred_with_next_feature = predictor(x_masked)
            loss_with_next_feature = loss_fn(pred_with_next_feature, y)

            # Value network loss.
            delta = loss_without_next_feature - loss_with_next_feature.detach()
            value_network_loss = nn.functional.mse_loss(
                pred_cmi[torch.arange(len(x)), actions], delta
            )

            # Calculate gradients.
            total_loss = torch.mean(value_network_loss) + torch.mean(
                loss_with_next_feature
            )
            step_losses.append(total_loss)

            # Updates.
            value_network_loss_total += value_network_loss.detach()
            pred_loss_total += loss_with_next_feature.mean().detach()
            loss_without_next_feature = loss_with_next_feature.detach()
            pred_without_next_feature = pred_with_next_feature.detach()

        # Each acquisition contributes independently because the
        # prediction used by the next acquisition is detached above.
        # Average the losses once, then run one backward pass per
        # optimizer update instead of one per acquisition.
        torch.stack(step_losses).mean().backward()

        self._value_losses.append(
            value_network_loss_total / settings.selections_per_episode
        )
        self._pred_losses.append(
            pred_loss_total / (settings.selections_per_episode + 1)
        )

    @override
    def validate_episode(self, episode: BatchEpisode) -> None:
        context = self.context
        predictor = context.predictor
        assert self._selection_costs is not None
        x = episode.features
        y = episode.label
        m_feat = episode.feature_mask
        m_sel = episode.selection_mask

        x_masked = apply_model_mask(context.mask_layer, x, m_feat)
        pred = predictor(x_masked)
        self._val_preds[0].append(pred)

        for i in range(1, context.settings.selections_per_episode + 1):
            # Estimate CMI using value network.
            x_masked = apply_model_mask(context.mask_layer, x, m_feat)
            pred_cmi = self._estimate_cmi(x_masked, pred)

            # Select next feature, ensure no repeats.
            pred_cmi -= 1e6 * m_sel
            best_feature_index = torch.argmax(
                pred_cmi / self._selection_costs, dim=1
            )
            m_sel = torch.max(
                m_sel, ind_to_onehot(best_feature_index, context.n_selections)
            )
            afa_selection = best_feature_index.to(torch.long).unsqueeze(1)
            m_feat = context.unmasker.unmask(
                masked_features=x_masked,
                feature_mask=m_feat.bool(),
                features=x,
                afa_selection=afa_selection,
                selection_mask=m_sel,
                feature_shape=context.feature_shape,
            )

            # Make prediction.
            x_masked = apply_model_mask(context.mask_layer, x, m_feat)
            pred = predictor(x_masked)
            self._val_preds[i].append(pred)

        self._val_targets.append(y)

    @override
    def end_epoch(self) -> EpochValidation:
        context = self.context
        loss_fn = context.settings.loss_fn
        train_value_loss = torch.stack(self._value_losses).mean()
        train_pred_loss = torch.stack(self._pred_losses).mean()

        y_val = torch.cat(self._val_targets)
        preds_cat = [torch.cat(p) for p in self._val_preds]
        val_pred_losses = [loss_fn(p, y_val).mean() for p in preds_cat]
        val_scores = [context.val_loss_fn(p, y_val) for p in preds_cat]
        val_loss_mean = torch.stack(val_pred_losses).mean()
        val_perf_mean = torch.stack(val_scores).mean()
        val_loss_final = val_pred_losses[-1]
        val_perf_final = val_scores[-1]

        return EpochValidation(
            score=val_perf_mean,
            metrics={
                "train_value_loss": float(train_value_loss.item()),
                "train_predictor_loss": float(train_pred_loss.item()),
                "val_loss_mean": float(val_loss_mean.mean().item()),
                "val_perf_mean": float(val_perf_mean.mean().item()),
                "val_loss_final": float(val_loss_final.mean().item()),
                "val_perf_final": float(val_perf_final.mean().item()),
                "epsilon": float(self.eps),
            },
        )

    @override
    def on_plateau(self, optimizer: Optimizer) -> bool:
        # Decay epsilon.
        self.eps = self.eps * self.eps_decay
        self._num_epsilon_steps += 1
        print(
            f"Decaying eps to {self.eps:.5f}, step = {self._num_epsilon_steps}"
        )

        # Early stopping.
        if self._num_epsilon_steps >= self.eps_steps:
            return True

        # Reset optimizer learning rate. Could fully reset optimizer and
        # scheduler, but this is simpler.
        for g in optimizer.param_groups:
            g["lr"] = self.context.settings.lr
        return False


__all__ = ["DIMEAFAMethod", "DIMETrainingObjective"]
