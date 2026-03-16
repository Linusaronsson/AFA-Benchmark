from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, final

import torch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from afabench.common.custom_types import (
        AFAAction,
        AFAMethod,
        AFAPredictFn,
        AFAUnmaskFn,
        FeatureMask,
        Features,
        Label,
        MaskedFeatures,
        SelectionMask,
    )

log = logging.getLogger(__name__)


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else 0.0


def _ensure_probabilities(predictions: torch.Tensor) -> torch.Tensor:
    """
    Normalize classifier outputs before using them as stop-risk estimates.

    `eval.py` already accepts either logits or probabilities for predictions.
    The shield needs calibrated class probabilities for
    `rho = 1 - max_y p(y | x_S, S)`, so we normalize here instead of assuming
    every classifier follows the same convention.
    """
    if predictions.numel() == 0:
        return predictions

    row_sums = predictions.sum(dim=-1)
    looks_like_probabilities = bool(
        torch.isfinite(predictions).all()
        and predictions.min().item() >= -1e-6
        and predictions.max().item() <= 1.0 + 1e-6
        and torch.allclose(
            row_sums,
            torch.ones_like(row_sums),
            atol=1e-4,
            rtol=1e-4,
        )
    )
    if looks_like_probabilities:
        return predictions
    return predictions.softmax(dim=-1)


@contextmanager
def _temporary_method_attr(
    afa_method: AFAMethod,
    attr_name: str,
    attr_value: object,
) -> Iterator[None]:
    previous_value = getattr(afa_method, attr_name)
    setattr(afa_method, attr_name, attr_value)
    try:
        yield
    finally:
        setattr(afa_method, attr_name, previous_value)


def _act_without_stop_action(
    afa_method: AFAMethod,
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    selection_mask: SelectionMask | None,
    label: Label | None,
    feature_shape: torch.Size | None,
) -> AFAAction:
    """
    Ask a method for a continuation action by temporarily disabling `STOP`.

    DIME-like methods expose a stop threshold, while AACO-like methods expose
    `force_acquisition`. Supporting those two hooks keeps the wrapper small
    while still covering the benchmark methods used here.
    """
    if hasattr(afa_method, "force_acquisition"):
        with _temporary_method_attr(afa_method, "force_acquisition", True):
            return afa_method.act(
                masked_features=masked_features,
                feature_mask=feature_mask,
                selection_mask=selection_mask,
                label=label,
                feature_shape=feature_shape,
            )

    if hasattr(afa_method, "lambda_threshold"):
        with _temporary_method_attr(afa_method, "lambda_threshold", -math.inf):
            return afa_method.act(
                masked_features=masked_features,
                feature_mask=feature_mask,
                selection_mask=selection_mask,
                label=label,
                feature_shape=feature_shape,
            )

    msg = (
        "Stop shielding only supports methods exposing either "
        "`force_acquisition` or `lambda_threshold`. "
        f"Got {type(afa_method).__name__}."
    )
    raise TypeError(msg)


def _compute_stop_risk(
    afa_predict_fn: AFAPredictFn,
    *,
    masked_features: MaskedFeatures,
    feature_mask: FeatureMask,
    label: Label | None,
    feature_shape: torch.Size | None,
) -> torch.Tensor:
    predictions = afa_predict_fn(
        masked_features=masked_features,
        feature_mask=feature_mask,
        label=label,
        feature_shape=feature_shape,
    )
    probabilities = _ensure_probabilities(predictions)
    return 1.0 - probabilities.max(dim=-1).values


@dataclass
@final
class StopShieldWrapper:
    """
    Deployment-time stop shield for AFA methods.

    The wrapper leaves the base policy unchanged except when it proposes
    `STOP` and the estimated stop risk `rho = 1 - max_y p(y | x_S, S)`
    exceeds `risk_threshold`.

    References:
    - Sheng, Parker, and Feng (2024), "Safe POMDP Online Planning via
      Shielding".
    - Moss et al. (2024), "ConstrainedZero: Chance-Constrained POMDP Planning
      using Learned Probabilistic Failure Surrogates and Adaptive Safety
      Constraints".
    """

    afa_method: AFAMethod
    afa_predict_fn: AFAPredictFn
    risk_threshold: float
    predictor_name: str
    requires_transition_context: bool = field(
        default=False,
        init=False,
        repr=False,
    )
    _n_steps: int = field(default=0, init=False, repr=False)
    _n_proposed_stops: int = field(default=0, init=False, repr=False)
    _n_rejected_stops: int = field(default=0, init=False, repr=False)
    _n_forced_continue_actions: int = field(default=0, init=False, repr=False)
    _n_unavoidable_stops: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 <= self.risk_threshold <= 1.0):
            msg = (
                "Expected stop-shield risk_threshold in [0, 1], got "
                f"{self.risk_threshold}."
            )
            raise ValueError(msg)

    def _compute_stop_risk(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None,
        feature_shape: torch.Size | None,
    ) -> torch.Tensor:
        return _compute_stop_risk(
            self.afa_predict_fn,
            masked_features=masked_features,
            feature_mask=feature_mask,
            label=label,
            feature_shape=feature_shape,
        )

    def _act_without_stop(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None,
        label: Label | None,
        feature_shape: torch.Size | None,
    ) -> AFAAction:
        return _act_without_stop_action(
            afa_method=self.afa_method,
            masked_features=masked_features,
            feature_mask=feature_mask,
            selection_mask=selection_mask,
            label=label,
            feature_shape=feature_shape,
        )

    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        action = self.afa_method.act(
            masked_features=masked_features,
            feature_mask=feature_mask,
            selection_mask=selection_mask,
            label=label,
            feature_shape=feature_shape,
        )
        assert action.ndim == 2, "Stop shielding expects rank-2 AFA actions."
        assert action.shape[-1] == 1, (
            "Stop shielding expects AFA actions with shape [batch, 1]."
        )

        stop_mask = action.squeeze(-1) == 0
        batch_size = action.shape[0]
        self._n_steps += batch_size
        self._n_proposed_stops += int(stop_mask.sum().item())
        if not stop_mask.any():
            return action

        stop_risk = self._compute_stop_risk(
            masked_features=masked_features,
            feature_mask=feature_mask,
            label=label,
            feature_shape=feature_shape,
        )
        reject_stop_mask = stop_mask & (stop_risk > self.risk_threshold)
        self._n_rejected_stops += int(reject_stop_mask.sum().item())
        if not reject_stop_mask.any():
            return action

        forced_action = self._act_without_stop(
            masked_features=masked_features[reject_stop_mask],
            feature_mask=feature_mask[reject_stop_mask],
            selection_mask=None
            if selection_mask is None
            else selection_mask[reject_stop_mask],
            label=None if label is None else label[reject_stop_mask],
            feature_shape=feature_shape,
        )
        forced_continue_mask = forced_action.squeeze(-1) != 0
        self._n_forced_continue_actions += int(
            forced_continue_mask.sum().item()
        )
        self._n_unavoidable_stops += int((~forced_continue_mask).sum().item())

        shielded_action = action.clone()
        shielded_action[reject_stop_mask] = forced_action
        return shielded_action

    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        return self(
            masked_features=masked_features,
            feature_mask=feature_mask,
            selection_mask=selection_mask,
            label=label,
            feature_shape=feature_shape,
        )

    def get_summary(self) -> dict[str, float | int | str]:
        return {
            "shield_type": "stop_risk_threshold",
            "predictor_name": self.predictor_name,
            "risk_threshold": self.risk_threshold,
            "n_steps": self._n_steps,
            "n_proposed_stops": self._n_proposed_stops,
            "n_rejected_stops": self._n_rejected_stops,
            "n_forced_continue_actions": self._n_forced_continue_actions,
            "n_unavoidable_stops": self._n_unavoidable_stops,
            "proposed_stop_rate": _safe_divide(
                self._n_proposed_stops, self._n_steps
            ),
            "rejected_stop_rate": _safe_divide(
                self._n_rejected_stops, self._n_steps
            ),
            "forced_continue_rate": _safe_divide(
                self._n_forced_continue_actions, self._n_steps
            ),
            "forced_continue_given_reject_rate": _safe_divide(
                self._n_forced_continue_actions, self._n_rejected_stops
            ),
        }


@dataclass
@final
class DualizedStopWrapper:
    """
    One-step dualized stop controller.

    The wrapper only intervenes when the base policy proposes `STOP`. It then
    compares:

    - `Q_stop ~= lambda * risk_hat(b)`
    - `Q_continue ~= c(a_base) + lambda * risk_hat(b')`

    where `a_base` is the forced continuation action and `b'` is the state
    reached after executing that action once. This is still myopic, but unlike
    a fixed `risk <= delta` threshold it can prefer early stopping on hard
    cases when the next acquisition is not worth the cost.

    The Lagrangian form mirrors constrained-POMDP planning, but this
    implementation intentionally stays one-step and cheap to evaluate.
    """

    afa_method: AFAMethod
    afa_predict_fn: AFAPredictFn
    afa_unmask_fn: AFAUnmaskFn
    predictor_name: str
    selection_costs: torch.Tensor
    dual_lambda: float
    requires_transition_context: bool = field(
        default=True,
        init=False,
        repr=False,
    )
    _n_steps: int = field(default=0, init=False, repr=False)
    _n_proposed_stops: int = field(default=0, init=False, repr=False)
    _n_rejected_stops: int = field(default=0, init=False, repr=False)
    _n_forced_continue_actions: int = field(default=0, init=False, repr=False)
    _n_unavoidable_stops: int = field(default=0, init=False, repr=False)
    _n_compared_stops: int = field(default=0, init=False, repr=False)
    _sum_stop_objective: float = field(default=0.0, init=False, repr=False)
    _sum_continue_objective: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dual_lambda < 0.0:
            msg = (
                f"Expected dualized stop lambda >= 0, got {self.dual_lambda}."
            )
            raise ValueError(msg)
        self.selection_costs = torch.as_tensor(
            self.selection_costs,
            dtype=torch.float32,
        )

    def _compute_stop_risk(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None,
        feature_shape: torch.Size | None,
    ) -> torch.Tensor:
        return _compute_stop_risk(
            self.afa_predict_fn,
            masked_features=masked_features,
            feature_mask=feature_mask,
            label=label,
            feature_shape=feature_shape,
        )

    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
        *,
        features: Features,
    ) -> AFAAction:
        action = self.afa_method.act(
            masked_features=masked_features,
            feature_mask=feature_mask,
            selection_mask=selection_mask,
            label=label,
            feature_shape=feature_shape,
        )
        assert action.ndim == 2, (
            "Dualized stopping expects rank-2 AFA actions."
        )
        assert action.shape[-1] == 1, (
            "Dualized stopping expects AFA actions with shape [batch, 1]."
        )

        stop_mask = action.squeeze(-1) == 0
        batch_size = action.shape[0]
        self._n_steps += batch_size
        self._n_proposed_stops += int(stop_mask.sum().item())
        if not stop_mask.any():
            return action

        stop_risk = self._compute_stop_risk(
            masked_features=masked_features,
            feature_mask=feature_mask,
            label=label,
            feature_shape=feature_shape,
        )
        stop_objective = self.dual_lambda * stop_risk

        forced_action = _act_without_stop_action(
            afa_method=self.afa_method,
            masked_features=masked_features[stop_mask],
            feature_mask=feature_mask[stop_mask],
            selection_mask=None
            if selection_mask is None
            else selection_mask[stop_mask],
            label=None if label is None else label[stop_mask],
            feature_shape=feature_shape,
        )
        forced_continue_mask = forced_action.squeeze(-1) != 0
        self._n_unavoidable_stops += int((~forced_continue_mask).sum().item())
        if not forced_continue_mask.any():
            return action

        stopped_indices = stop_mask.nonzero(as_tuple=True)[0]
        continue_indices = stopped_indices[forced_continue_mask]
        forced_continue_action = forced_action[forced_continue_mask]
        forced_selection = forced_continue_action.squeeze(-1) - 1

        assert selection_mask is not None, (
            "Dualized stop control requires selection masks during evaluation."
        )
        selection_mask_subset = selection_mask[continue_indices]
        updated_feature_mask = self.afa_unmask_fn(
            masked_features=masked_features[continue_indices],
            feature_mask=feature_mask[continue_indices],
            features=features[continue_indices],
            afa_selection=forced_selection,
            selection_mask=selection_mask_subset,
            label=None if label is None else label[continue_indices],
            feature_shape=feature_shape,
        )
        updated_masked_features = features[continue_indices].clone()
        updated_masked_features[~updated_feature_mask] = 0.0
        continuation_risk = self._compute_stop_risk(
            masked_features=updated_masked_features,
            feature_mask=updated_feature_mask,
            label=None if label is None else label[continue_indices],
            feature_shape=feature_shape,
        )
        action_costs = self.selection_costs.to(forced_selection.device)[
            forced_selection
        ]
        continuation_objective = action_costs + (
            self.dual_lambda * continuation_risk
        )
        stop_objective_subset = stop_objective[continue_indices]
        self._n_compared_stops += int(continue_indices.numel())
        self._sum_stop_objective += float(stop_objective_subset.sum().item())
        self._sum_continue_objective += float(
            continuation_objective.sum().item()
        )

        reject_stop_mask = stop_objective_subset > continuation_objective
        self._n_rejected_stops += int(reject_stop_mask.sum().item())
        self._n_forced_continue_actions += int(reject_stop_mask.sum().item())
        if not reject_stop_mask.any():
            return action

        shielded_action = action.clone()
        shielded_action[continue_indices[reject_stop_mask]] = (
            forced_continue_action[reject_stop_mask]
        )
        return shielded_action

    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
        *,
        features: Features,
    ) -> AFAAction:
        return self(
            masked_features=masked_features,
            feature_mask=feature_mask,
            selection_mask=selection_mask,
            label=label,
            feature_shape=feature_shape,
            features=features,
        )

    def get_summary(self) -> dict[str, float | int | str]:
        compared_count = max(self._n_compared_stops, 1)
        return {
            "shield_type": "dualized_stop",
            "predictor_name": self.predictor_name,
            "dual_lambda": self.dual_lambda,
            "n_steps": self._n_steps,
            "n_proposed_stops": self._n_proposed_stops,
            "n_rejected_stops": self._n_rejected_stops,
            "n_forced_continue_actions": self._n_forced_continue_actions,
            "n_unavoidable_stops": self._n_unavoidable_stops,
            "n_compared_stops": self._n_compared_stops,
            "proposed_stop_rate": _safe_divide(
                self._n_proposed_stops, self._n_steps
            ),
            "rejected_stop_rate": _safe_divide(
                self._n_rejected_stops, self._n_steps
            ),
            "forced_continue_rate": _safe_divide(
                self._n_forced_continue_actions, self._n_steps
            ),
            "avg_stop_objective": self._sum_stop_objective / compared_count,
            "avg_continue_objective": self._sum_continue_objective
            / compared_count,
        }
