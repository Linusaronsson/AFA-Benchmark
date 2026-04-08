from __future__ import annotations

from typing import ClassVar, cast, final, override

import torch

from afabench.common.custom_types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
    SelectionMask,
)


@final
class XORNoisyShortcutInitializer(AFAInitializer):
    """
    Cold-start training initializer for the XOR noisy-shortcut benchmark.

    The episode always starts with no observed features. The initializer only
    restricts train-time support by forbidding feature x2 according to one of
    three mechanisms:
      - MCAR: independent Bernoulli blocking
      - MAR: x2 blocking depends on x1
      - MNAR: x2 blocking depends on x2 itself
    """

    SUPPORTED_MECHANISMS: ClassVar[set[str]] = {"mcar", "mar", "mnar"}

    def __init__(
        self,
        *,
        mechanism: str,
        p: float,
        structured_shift: float = 0.25,
        blocked_feature_idx: int = 1,
    ):
        if mechanism not in self.SUPPORTED_MECHANISMS:
            msg = (
                f"Unknown mechanism: {mechanism}. "
                f"Supported: {sorted(self.SUPPORTED_MECHANISMS)}"
            )
            raise ValueError(msg)
        if not (0.0 <= p <= 1.0):
            msg = f"p must be in [0, 1], got {p}."
            raise ValueError(msg)
        if not (0.0 <= structured_shift <= 0.5):
            msg = (
                "structured_shift must be in [0, 0.5], got "
                f"{structured_shift}."
            )
            raise ValueError(msg)

        self.mechanism = mechanism
        self.p = p
        self.structured_shift = structured_shift
        self.blocked_feature_idx = blocked_feature_idx
        self.seed: int | None = None
        self._rng: torch.Generator | None = None
        self._last_training_forbidden_mask: FeatureMask | None = None
        self._last_forbidden_selection_mask: SelectionMask | None = None

    def _get_rng(self) -> torch.Generator:
        if self._rng is None:
            self._rng = torch.Generator()
            if self.seed is not None:
                self._rng.manual_seed(self.seed)
        return self._rng

    def _structured_delta(self) -> float:
        if self.p in {0.0, 1.0}:
            return 0.0
        return min(
            self.structured_shift,
            max(self.p - 1e-3, 0.0),
            max(1.0 - self.p - 1e-3, 0.0),
        )

    @override
    def set_seed(self, seed: int | None) -> None:
        self.seed = seed
        self._rng = None

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        del label
        assert feature_shape is not None, (
            "feature_shape must be provided for XORNoisyShortcutInitializer"
        )
        assert feature_shape.numel() == 3, (
            "XORNoisyShortcutInitializer expects exactly three features, "
            f"got {feature_shape}."
        )

        flat_features = features.reshape(-1, feature_shape.numel())
        batch_shape = features.shape[: -len(feature_shape)]
        batch_size = flat_features.shape[0]
        device = flat_features.device

        block_probs = self._block_probabilities(flat_features)
        blocked = (
            torch.rand((batch_size,), generator=self._get_rng()).to(device)
            < block_probs
        )

        observed = torch.zeros_like(flat_features, dtype=torch.bool)
        training_forbidden = torch.zeros_like(flat_features, dtype=torch.bool)
        training_forbidden[:, self.blocked_feature_idx] = blocked

        forbidden_selection = torch.zeros(
            (batch_size, feature_shape.numel()),
            dtype=torch.bool,
            device=device,
        )
        forbidden_selection[:, self.blocked_feature_idx] = blocked

        self._last_training_forbidden_mask = training_forbidden.reshape(
            batch_shape + feature_shape
        )
        self._last_forbidden_selection_mask = forbidden_selection.reshape(
            batch_shape + torch.Size((feature_shape.numel(),))
        )
        return observed.reshape(batch_shape + feature_shape)

    def _block_probabilities(
        self,
        flat_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = flat_features.shape[0]
        device = flat_features.device
        base = torch.full(
            (batch_size,),
            self.p,
            dtype=torch.float32,
            device=device,
        )
        if self.mechanism == "mcar":
            return base

        delta = self._structured_delta()
        high = self.p + delta
        low = self.p - delta

        if self.mechanism == "mar":
            driver = flat_features[:, 0] < 0.5
        elif self.mechanism == "mnar":
            driver = flat_features[:, self.blocked_feature_idx] < 0.5
        else:
            msg = f"Unknown mechanism: {self.mechanism}"
            raise ValueError(msg)

        return torch.where(driver, high, low).to(device=device)

    @override
    def get_training_forbidden_mask(
        self,
        observed_mask: FeatureMask,
    ) -> FeatureMask:
        if self._last_training_forbidden_mask is not None:
            return self._last_training_forbidden_mask
        return torch.zeros_like(observed_mask, dtype=torch.bool)

    def get_forbidden_selection_mask(
        self,
        _observed_mask: FeatureMask,
        feature_shape: torch.Size,
    ) -> SelectionMask:
        if self._last_forbidden_selection_mask is None:
            msg = (
                "XORNoisyShortcutInitializer has no cached forbidden "
                "selection mask. Call initialize() first."
            )
            raise RuntimeError(msg)

        expected_shape = feature_shape.numel()
        last_shape = self._last_forbidden_selection_mask.shape[-1]
        assert last_shape == expected_shape, (
            "Cached forbidden selection mask does not match feature shape: "
            f"{last_shape} != {expected_shape}."
        )
        return cast("SelectionMask", self._last_forbidden_selection_mask)
