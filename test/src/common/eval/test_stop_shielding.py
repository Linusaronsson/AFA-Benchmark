from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast, final, override

import pytest
import torch

from afabench.afa_rl.common.afa_methods import RLAFAMethod
from afabench.common.unmaskers.direct_unmasker import DirectUnmasker
from afabench.eval.stop_shielding import DualizedStopWrapper, StopShieldWrapper

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tensordict import TensorDict
    from tensordict.nn import TensorDictModuleBase

    from afabench.common.custom_types import AFAMethod


class _MaskedGreedyPolicy(torch.nn.Module):
    @override
    def forward(self, td: TensorDict) -> TensorDict:
        allowed_action_mask = td["allowed_action_mask"].bool()
        actions = torch.zeros(
            allowed_action_mask.shape[0],
            dtype=torch.long,
            device=allowed_action_mask.device,
        )
        for row_idx, row_mask in enumerate(allowed_action_mask):
            allowed_actions = row_mask.nonzero(as_tuple=False).flatten()
            if allowed_actions.numel() > 0:
                actions[row_idx] = allowed_actions[0]
        td["action"] = actions
        return td


@final
class _ConstantClassifier:
    n_classes: int
    _device: torch.device

    def __init__(
        self,
        n_classes: int = 2,
        device: torch.device | None = None,
    ) -> None:
        self.n_classes = n_classes
        self._device = torch.device("cpu") if device is None else device

    def __call__(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,
        label: torch.Tensor | None = None,
        feature_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        del feature_mask, label, feature_shape
        probs = torch.zeros(
            (masked_features.shape[0], self.n_classes),
            dtype=torch.float32,
            device=masked_features.device,
        )
        probs[:, 0] = 0.6
        probs[:, 1] = 0.4
        return probs

    def save(self, path: Path) -> None:
        torch.save({"n_classes": self.n_classes}, path)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        state = torch.load(path, map_location=device)
        return cls(n_classes=state["n_classes"], device=device)

    def to(self, device: torch.device) -> Self:
        self._device = device
        return self

    @property
    def device(self) -> torch.device:
        return self._device


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


class ForceAcquireFirstMethod:
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
        action = 1 if self.force_acquisition else 0
        return torch.full(
            (masked_features.shape[0], 1),
            action,
            dtype=torch.long,
        )


class UnavoidableStopMethod:
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
        return torch.zeros((masked_features.shape[0], 1), dtype=torch.long)


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


def _feature_mask_risk_predict_fn(
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
    label: torch.Tensor | None = None,
    feature_shape: torch.Size | None = None,
) -> torch.Tensor:
    del masked_features, label, feature_shape
    confident = feature_mask[:, 0]
    probs = torch.empty((feature_mask.shape[0], 2), dtype=torch.float32)
    probs[confident] = torch.tensor([0.95, 0.05])
    probs[~confident] = torch.tensor([0.55, 0.45])
    return probs


def _first_feature_unmask(
    masked_features: torch.Tensor,
    feature_mask: torch.Tensor,
    features: torch.Tensor,
    afa_selection: torch.Tensor,
    selection_mask: torch.Tensor | None = None,
    label: torch.Tensor | None = None,
    feature_shape: torch.Size | None = None,
) -> torch.Tensor:
    del masked_features, features, selection_mask, label, feature_shape
    new_feature_mask = feature_mask.clone()
    batch_indices = torch.arange(feature_mask.shape[0])
    new_feature_mask[batch_indices, afa_selection.view(-1)] = True
    return new_feature_mask


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


def test_stop_shield_supports_rl_methods_via_force_acquisition() -> None:
    method = RLAFAMethod(
        policy_tdmodule=cast(
            "TensorDictModuleBase",
            cast("object", _MaskedGreedyPolicy()),
        ),
        afa_classifier=_ConstantClassifier(),
        _device=torch.device("cpu"),
    )
    shield = StopShieldWrapper(
        afa_method=cast("AFAMethod", cast("object", method)),
        afa_predict_fn=_fixed_predict_fn(torch.tensor([[0.6, 0.4]])),
        risk_threshold=0.2,
        predictor_name="fixed",
    )

    actions = shield.act(
        masked_features=torch.zeros((1, 2)),
        feature_mask=torch.zeros((1, 2), dtype=torch.bool),
        selection_mask=torch.tensor([[False, True]], dtype=torch.bool),
    )

    assert torch.equal(actions, torch.tensor([[1]]))
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


def test_dualized_stop_wrapper_lambda_zero_prefers_stop() -> None:
    wrapper = DualizedStopWrapper(
        afa_method=cast(
            "AFAMethod", cast("object", ForceAcquireFirstMethod())
        ),
        afa_predict_fn=_feature_mask_risk_predict_fn,
        afa_unmask_fn=_first_feature_unmask,
        predictor_name="mask-risk",
        selection_costs=torch.tensor([0.1]),
        dual_lambda=0.0,
    )

    actions = wrapper(
        masked_features=torch.zeros((1, 1)),
        feature_mask=torch.zeros((1, 1), dtype=torch.bool),
        selection_mask=torch.zeros((1, 1), dtype=torch.bool),
        features=torch.ones((1, 1)),
    )

    assert torch.equal(actions, torch.tensor([[0]]))
    assert wrapper.get_summary()["n_rejected_stops"] == 0


def test_dualized_stop_wrapper_rejects_risky_stop_when_continue_is_better() -> (
    None
):
    wrapper = DualizedStopWrapper(
        afa_method=cast(
            "AFAMethod", cast("object", ForceAcquireFirstMethod())
        ),
        afa_predict_fn=_feature_mask_risk_predict_fn,
        afa_unmask_fn=_first_feature_unmask,
        predictor_name="mask-risk",
        selection_costs=torch.tensor([0.1]),
        dual_lambda=2.0,
    )

    actions = wrapper(
        masked_features=torch.zeros((1, 1)),
        feature_mask=torch.zeros((1, 1), dtype=torch.bool),
        selection_mask=torch.zeros((1, 1), dtype=torch.bool),
        features=torch.ones((1, 1)),
    )

    assert torch.equal(actions, torch.tensor([[1]]))
    assert wrapper.get_summary()["n_rejected_stops"] == 1
    assert wrapper.get_summary()["n_forced_continue_actions"] == 1


def test_dualized_stop_wrapper_keeps_stop_when_no_legal_continue_exists() -> (
    None
):
    wrapper = DualizedStopWrapper(
        afa_method=cast("AFAMethod", cast("object", UnavoidableStopMethod())),
        afa_predict_fn=_feature_mask_risk_predict_fn,
        afa_unmask_fn=_first_feature_unmask,
        predictor_name="mask-risk",
        selection_costs=torch.tensor([0.1]),
        dual_lambda=10.0,
    )

    actions = wrapper(
        masked_features=torch.zeros((1, 1)),
        feature_mask=torch.zeros((1, 1), dtype=torch.bool),
        selection_mask=torch.zeros((1, 1), dtype=torch.bool),
        features=torch.ones((1, 1)),
    )

    assert torch.equal(actions, torch.tensor([[0]]))
    assert wrapper.get_summary()["n_unavoidable_stops"] == 1


def test_dualized_stop_wrapper_handles_direct_unmasker_batch_of_one() -> None:
    wrapper = DualizedStopWrapper(
        afa_method=cast(
            "AFAMethod", cast("object", ForceAcquireFirstMethod())
        ),
        afa_predict_fn=_feature_mask_risk_predict_fn,
        afa_unmask_fn=DirectUnmasker().unmask,
        predictor_name="mask-risk",
        selection_costs=torch.tensor([0.1, 0.1]),
        dual_lambda=2.0,
    )

    actions = wrapper(
        masked_features=torch.zeros((1, 2)),
        feature_mask=torch.zeros((1, 2), dtype=torch.bool),
        selection_mask=torch.zeros((1, 2), dtype=torch.bool),
        features=torch.ones((1, 2)),
        feature_shape=torch.Size([2]),
    )

    assert torch.equal(actions, torch.tensor([[1]]))
