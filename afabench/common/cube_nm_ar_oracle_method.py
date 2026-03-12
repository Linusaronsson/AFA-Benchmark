from __future__ import annotations

from typing import TYPE_CHECKING, Self, final, override

import torch

from afabench.common.custom_types import (
    AFAAction,
    AFAMethod,
    FeatureMask,
    Label,
    MaskedFeatures,
    SelectionMask,
)

if TYPE_CHECKING:
    from pathlib import Path


@final
class CubeNMAROracleMethod(AFAMethod):
    """Deterministic acquisition oracle for CUBE-NM-AR."""

    _class_version = "3.0.0"
    _state_keys = frozenset(
        {
            "n_contexts",
            "n_safe_contexts",
            "block_size",
            "n_classes",
            "context_action_cost",
            "selectable_feature_costs",
            "max_cost",
        }
    )

    def __init__(
        self,
        *,
        n_contexts: int,
        n_safe_contexts: int,
        block_size: int,
        n_classes: int,
        context_action_cost: float,
        selectable_feature_costs: torch.Tensor,
        device: torch.device | None = None,
        max_cost: float | None = None,
    ):
        self.n_contexts = n_contexts
        self.n_safe_contexts = n_safe_contexts
        self.block_size = block_size
        self.n_classes = n_classes
        self.context_action_cost = float(context_action_cost)
        self.selectable_feature_costs = (
            selectable_feature_costs.float().clone()
        )
        self.max_cost = None if max_cost is None else float(max_cost)
        self._device = device or torch.device("cpu")
        self.selectable_feature_costs = self.selectable_feature_costs.to(
            self._device
        )

    @property
    def _rescue_action(self) -> int:
        return 2 + self.n_contexts * self.block_size

    def _infer_context_index(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
    ) -> torch.Tensor:
        batch_size = masked_features.shape[0]
        context_idx = torch.full(
            (batch_size,),
            -1,
            dtype=torch.long,
            device=masked_features.device,
        )

        context_mask = feature_mask[:, : self.n_contexts]
        context_observed = context_mask.any(dim=1)
        if context_observed.any():
            context_idx[context_observed] = masked_features[
                context_observed, : self.n_contexts
            ].argmax(dim=1)

        return context_idx

    def _current_cost(self, feature_mask: FeatureMask) -> torch.Tensor:
        context_acquired = (
            feature_mask[:, : self.n_contexts].any(dim=1).float()
        )
        selectable_feature_mask = feature_mask[:, self.n_contexts :].float()
        return context_acquired * self.context_action_cost + (
            selectable_feature_mask
            * self.selectable_feature_costs.unsqueeze(0)
        ).sum(dim=1)

    def _action_cost(self, action: int) -> float:
        if action == 1:
            return self.context_action_cost
        if action <= 0:
            return 0.0
        return float(self.selectable_feature_costs[action - 2].item())

    def _would_exceed_cost(self, current_cost: float, action: int) -> bool:
        if self.max_cost is None or action == 0:
            return False
        return current_cost + self._action_cost(action) > self.max_cost

    def _planned_actions_for_context(self, context_idx: int) -> list[int]:
        block_start_action = 2 + context_idx * self.block_size
        if context_idx < self.n_safe_contexts:
            return [
                block_start_action,
                block_start_action + 1,
                block_start_action + 2,
            ]
        return [
            block_start_action,
            block_start_action + 1,
            self._rescue_action,
        ]

    @override
    def act(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        selection_mask: SelectionMask | None = None,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> AFAAction:
        del label, feature_shape
        assert selection_mask is not None, (
            "CubeNMAROracleMethod requires selection_mask."
        )
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)
        selection_mask = selection_mask.to(self._device)

        context_idx = self._infer_context_index(masked_features, feature_mask)
        current_cost = self._current_cost(feature_mask)

        actions = torch.zeros(
            (masked_features.shape[0], 1),
            dtype=torch.long,
            device=self._device,
        )

        for sample_idx in range(masked_features.shape[0]):
            sample_context_idx = int(context_idx[sample_idx].item())
            if sample_context_idx < 0:
                if not selection_mask[sample_idx, 0]:
                    candidate_action = 1
                    if not self._would_exceed_cost(
                        float(current_cost[sample_idx].item()),
                        candidate_action,
                    ):
                        actions[sample_idx, 0] = candidate_action
                continue

            for candidate_action in self._planned_actions_for_context(
                sample_context_idx
            ):
                if selection_mask[sample_idx, candidate_action - 1]:
                    continue
                if self._would_exceed_cost(
                    float(current_cost[sample_idx].item()),
                    candidate_action,
                ):
                    break
                actions[sample_idx, 0] = candidate_action
                break

        return actions.to(original_device)

    @override
    def predict(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        del feature_mask, label, feature_shape
        batch_size = masked_features.shape[0]
        return torch.full(
            (batch_size, self.n_classes),
            1.0 / self.n_classes,
            dtype=masked_features.dtype,
            device=masked_features.device,
        )

    @override
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "n_contexts": self.n_contexts,
                "n_safe_contexts": self.n_safe_contexts,
                "block_size": self.block_size,
                "n_classes": self.n_classes,
                "context_action_cost": self.context_action_cost,
                "selectable_feature_costs": self.selectable_feature_costs.cpu(),
                "max_cost": self.max_cost,
            },
            path / "method.pt",
        )

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        data = torch.load(path / "method.pt", map_location=device)
        state_keys = set(data.keys())
        if state_keys != set(cls._state_keys):
            missing = sorted(cls._state_keys - state_keys)
            unexpected = sorted(state_keys - cls._state_keys)
            msg = (
                "CubeNMAROracleMethod state does not match the simplified "
                f"schema. Missing keys: {missing}; unexpected keys: "
                f"{unexpected}."
            )
            raise KeyError(msg)
        return cls(
            n_contexts=data["n_contexts"],
            n_safe_contexts=data["n_safe_contexts"],
            block_size=data["block_size"],
            n_classes=data["n_classes"],
            context_action_cost=data["context_action_cost"],
            selectable_feature_costs=data["selectable_feature_costs"],
            device=device,
            max_cost=data["max_cost"],
        )

    @override
    def to(self, device: torch.device) -> Self:
        self._device = device
        self.selectable_feature_costs = self.selectable_feature_costs.to(
            device
        )
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device

    @property
    @override
    def has_builtin_classifier(self) -> bool:
        return False

    @property
    @override
    def cost_param(self) -> float | None:
        return self.max_cost

    @override
    def set_cost_param(self, cost_param: float) -> None:
        self.max_cost = float(cost_param)
