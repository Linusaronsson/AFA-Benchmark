from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
import torch

from afabench.eval.stop_shielding import StopShieldWrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    from afabench.common.custom_types import AFAMethod


class ThresholdStopMethod:
    def __init__(self) -> None:
        self.lambda_threshold: float = 0.5

    def act(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,
        selection_mask: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del feature_mask, selection_mask, label, feature_shape
        action = 0 if self.lambda_threshold > -1e10 else 2
        return torch.full(
            (masked_features.shape[0], 1), action, dtype=torch.long
        )


class ForceAcquisitionMethod:
    def __init__(self) -> None:
        self.force_acquisition: bool = False

    def act(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,
        selection_mask: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del feature_mask, selection_mask, label, feature_shape
        action = 3 if self.force_acquisition else 0
        return torch.full(
            (masked_features.shape[0], 1), action, dtype=torch.long
        )


class UnsupportedStopMethod:
    def act(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,
        selection_mask: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del masked_features, feature_mask, selection_mask, label, feature_shape
        return torch.zeros((1, 1), dtype=torch.long)


def _fixed_predict_fn(
    predictions: torch.Tensor,
) -> Callable[..., torch.Tensor]:
    def predict(
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,
        label: torch.Tensor | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del feature_mask, label, feature_shape
        return predictions[: masked_features.shape[0]]

    return predict


def test_stop_shield_rejects_only_risky_stops_for_threshold_method() -> None:
    method = ThresholdStopMethod()
    shield = StopShieldWrapper(
        afa_method=cast("AFAMethod", cast("object", method)),
        afa_predict_fn=_fixed_predict_fn(
            torch.tensor([[0.55, 0.45], [0.95, 0.05]])
        ),
        risk_threshold=0.2,
        predictor_name="fixed",
    )

    actions = shield.act(
        masked_features=torch.zeros((2, 4)),
        feature_mask=torch.zeros((2, 4), dtype=torch.bool),
        selection_mask=torch.zeros((2, 4), dtype=torch.bool),
    )

    assert torch.equal(actions, torch.tensor([[2], [0]]))
    assert method.lambda_threshold == 0.5
    assert shield.get_summary()["n_rejected_stops"] == 1
    assert shield.get_summary()["n_forced_continue_actions"] == 1


def test_stop_shield_uses_force_acquisition_when_available() -> None:
    method = ForceAcquisitionMethod()
    shield = StopShieldWrapper(
        afa_method=cast("AFAMethod", cast("object", method)),
        afa_predict_fn=_fixed_predict_fn(torch.tensor([[0.6, 0.4]])),
        risk_threshold=0.2,
        predictor_name="fixed",
    )

    actions = shield.act(
        masked_features=torch.zeros((1, 3)),
        feature_mask=torch.zeros((1, 3), dtype=torch.bool),
        selection_mask=torch.zeros((1, 3), dtype=torch.bool),
    )

    assert torch.equal(actions, torch.tensor([[3]]))
    assert method.force_acquisition is False


def test_stop_shield_accepts_low_risk_stop_from_logits() -> None:
    method = ThresholdStopMethod()
    shield = StopShieldWrapper(
        afa_method=cast("AFAMethod", cast("object", method)),
        afa_predict_fn=_fixed_predict_fn(torch.tensor([[3.0, 0.0]])),
        risk_threshold=0.1,
        predictor_name="fixed",
    )

    actions = shield.act(
        masked_features=torch.zeros((1, 2)),
        feature_mask=torch.zeros((1, 2), dtype=torch.bool),
    )

    assert torch.equal(actions, torch.tensor([[0]]))
    assert shield.get_summary()["n_rejected_stops"] == 0


def test_stop_shield_raises_for_unsupported_method() -> None:
    shield = StopShieldWrapper(
        afa_method=cast(
            "AFAMethod",
            cast("object", UnsupportedStopMethod()),
        ),
        afa_predict_fn=_fixed_predict_fn(torch.tensor([[0.6, 0.4]])),
        risk_threshold=0.2,
        predictor_name="fixed",
    )

    with pytest.raises(
        TypeError, match="Stop shielding only supports methods"
    ):
        shield.act(
            masked_features=torch.zeros((1, 2)),
            feature_mask=torch.zeros((1, 2), dtype=torch.bool),
        )
