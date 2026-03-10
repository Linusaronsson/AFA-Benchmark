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
        FeatureMask,
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


@dataclass
@final
class StopShieldWrapper:
    """
    Deployment-time stop shield for AFA methods.

    Source idea:
    - stop-risk constraint in `deliverables/thesis/Typst/include/chapters/theory.typ`
      (`rho(x_S, S) <= delta` before `STOP`)
    - shielding-style safe POMDP planning, where a wrapper blocks unsafe actions
      while leaving the base policy unchanged.

    We keep the implementation narrow on purpose: only `STOP` is shielded,
    because in AFA the main safety question is usually *when to stop* rather
    than whether feature acquisition itself is unsafe.
    """

    afa_method: AFAMethod
    afa_predict_fn: AFAPredictFn
    risk_threshold: float
    predictor_name: str
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
        predictions = self.afa_predict_fn(
            masked_features=masked_features,
            feature_mask=feature_mask,
            label=label,
            feature_shape=feature_shape,
        )
        probabilities = _ensure_probabilities(predictions)
        return 1.0 - probabilities.max(dim=-1).values

    def _act_without_stop(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None,
        label: Label | None,
        feature_shape: torch.Size | None,
    ) -> AFAAction:
        # DIME-like methods expose a stop threshold, while AACO-like methods
        # expose an explicit "must acquire something" flag. Supporting those two
        # hooks gives us a small shield that still covers the main local methods.
        if hasattr(self.afa_method, "force_acquisition"):
            with _temporary_method_attr(
                self.afa_method, "force_acquisition", True
            ):
                return self.afa_method.act(
                    masked_features=masked_features,
                    feature_mask=feature_mask,
                    selection_mask=selection_mask,
                    label=label,
                    feature_shape=feature_shape,
                )

        if hasattr(self.afa_method, "lambda_threshold"):
            with _temporary_method_attr(
                self.afa_method, "lambda_threshold", -math.inf
            ):
                return self.afa_method.act(
                    masked_features=masked_features,
                    feature_mask=feature_mask,
                    selection_mask=selection_mask,
                    label=label,
                    feature_shape=feature_shape,
                )

        msg = (
            "Stop shielding only supports methods exposing either "
            "`force_acquisition` or `lambda_threshold`. "
            f"Got {type(self.afa_method).__name__}."
        )
        raise TypeError(msg)

    def act(
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
