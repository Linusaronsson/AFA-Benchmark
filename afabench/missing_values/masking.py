"""Fitted MCAR, MAR, and MNAR mechanisms for training-data views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import torch

if TYPE_CHECKING:
    from afabench.missing_values.config import TrainingMissingnessConfig

_MAX_NONEMPTY_RESAMPLES = 1024
_BISECTION_STEPS = 80


def _generator(seed: int, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _fit_intercepts(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Fit one intercept per column so mean sigmoid probability is ``p``."""
    if logits.numel() == 0:
        return torch.empty(
            logits.shape[-1], dtype=logits.dtype, device=logits.device
        )
    low = torch.full(
        (logits.shape[1],), -80.0, dtype=logits.dtype, device=logits.device
    )
    high = torch.full_like(low, 80.0)
    for _ in range(_BISECTION_STEPS):
        mid = (low + high) / 2
        prevalence = torch.sigmoid(logits + mid).mean(dim=0)
        low = torch.where(prevalence < p, mid, low)
        high = torch.where(prevalence >= p, mid, high)
    return (low + high) / 2


def _scaled_coefficients(
    inputs: torch.Tensor,
    n_outputs: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    coefficients = torch.randn(
        inputs.shape[1],
        n_outputs,
        generator=generator,
        dtype=inputs.dtype,
        device=inputs.device,
    )
    projected = inputs @ coefficients
    eps = torch.finfo(inputs.dtype).eps
    scale = projected.std(dim=0, unbiased=False).clamp_min(eps)
    return coefficients / scale


@dataclass
class FittedMissingnessMechanism:
    """A mechanism whose feature choices and logistic parameters are fixed."""

    config: TrainingMissingnessConfig
    input_indices: torch.Tensor
    target_indices: torch.Tensor
    coefficients: torch.Tensor
    intercepts: torch.Tensor

    @classmethod
    def fit(
        cls,
        features: torch.Tensor,
        config: TrainingMissingnessConfig,
        *,
        seed: int,
    ) -> Self:
        if features.ndim != 2:
            msg = "Training missingness currently requires flat tabular data."
            raise ValueError(msg)
        if not 0.0 <= config.p < 1.0:
            msg = f"p must be in [0, 1), got {config.p}."
            raise ValueError(msg)
        if config.mechanism not in {
            "none",
            "mcar",
            "mar",
            "mnar_logistic",
            "mnar_self",
        }:
            msg = f"Unsupported missingness mechanism: {config.mechanism}"
            raise ValueError(msg)

        x = features.detach().to(dtype=torch.float64)
        n_features = x.shape[1]
        rng = _generator(seed, x.device)
        all_indices = torch.arange(n_features, device=x.device)

        if config.mechanism in {"none", "mcar"}:
            empty = torch.empty(0, dtype=torch.long, device=x.device)
            return cls(
                config=config,
                input_indices=empty,
                target_indices=all_indices,
                coefficients=torch.empty(
                    0, n_features, dtype=x.dtype, device=x.device
                ),
                intercepts=torch.empty(
                    n_features, dtype=x.dtype, device=x.device
                ),
            )

        if config.mechanism == "mnar_self":
            coefficients = torch.randn(
                n_features, generator=rng, dtype=x.dtype, device=x.device
            )
            projected = x * coefficients
            eps = torch.finfo(x.dtype).eps
            coefficients = coefficients / projected.std(
                dim=0, unbiased=False
            ).clamp_min(eps)
            logits = x * coefficients
            return cls(
                config=config,
                input_indices=all_indices,
                target_indices=all_indices,
                coefficients=coefficients,
                intercepts=_fit_intercepts(logits, config.p),
            )

        fraction = (
            config.p_obs if config.mechanism == "mar" else config.p_params
        )
        if not 0.0 < fraction <= 1.0:
            msg = (
                f"Mechanism input fraction must be in (0, 1], got {fraction}."
            )
            raise ValueError(msg)
        n_inputs = min(max(int(fraction * n_features), 1), n_features)
        input_indices = torch.randperm(
            n_features, generator=rng, device=x.device
        )[:n_inputs]
        if config.mechanism == "mnar_logistic" and not config.exclude_inputs:
            target_indices = all_indices
        else:
            input_set = set(input_indices.tolist())
            target_indices = torch.tensor(
                [idx for idx in range(n_features) if idx not in input_set],
                dtype=torch.long,
                device=x.device,
            )

        coefficients = _scaled_coefficients(
            x[:, input_indices], len(target_indices), generator=rng
        )
        logits = x[:, input_indices] @ coefficients
        return cls(
            config=config,
            input_indices=input_indices,
            target_indices=target_indices,
            coefficients=coefficients,
            intercepts=_fit_intercepts(logits, config.p),
        )

    def _sample_observed_once(
        self, features: torch.Tensor, *, seed: int
    ) -> torch.Tensor:
        x = features.detach().to(dtype=torch.float64)
        observed = torch.ones_like(x, dtype=torch.bool)
        if self.config.mechanism == "none" or self.config.p == 0.0:
            return observed

        rng = _generator(seed, x.device)
        if self.config.mechanism == "mcar":
            missing = (
                torch.rand(
                    x.shape, generator=rng, dtype=x.dtype, device=x.device
                )
                < self.config.p
            )
            return ~missing

        if self.config.mechanism == "mnar_self":
            logits = x * self.coefficients.to(x.device)
        else:
            logits = x[
                :, self.input_indices.to(x.device)
            ] @ self.coefficients.to(x.device)
        probabilities = torch.sigmoid(logits + self.intercepts.to(x.device))
        missing = (
            torch.rand(
                probabilities.shape,
                generator=rng,
                dtype=probabilities.dtype,
                device=x.device,
            )
            < probabilities
        )
        observed[:, self.target_indices.to(x.device)] = ~missing
        if (
            self.config.mechanism == "mnar_logistic"
            and self.config.exclude_inputs
        ):
            input_missing = (
                torch.rand(
                    (x.shape[0], len(self.input_indices)),
                    generator=rng,
                    dtype=x.dtype,
                    device=x.device,
                )
                < self.config.p
            )
            observed[:, self.input_indices.to(x.device)] = ~input_missing
        return observed

    def sample_observed_mask(
        self, features: torch.Tensor, *, seed: int
    ) -> torch.Tensor:
        """Sample a deterministic mask and reject fully missing instances."""
        observed = self._sample_observed_once(features, seed=seed)
        bad = ~observed.any(dim=1)
        attempt = 0
        while bad.any() and attempt < _MAX_NONEMPTY_RESAMPLES:
            attempt += 1
            observed[bad] = self._sample_observed_once(
                features[bad], seed=seed + attempt
            )
            bad = ~observed.any(dim=1)
        if bad.any():
            msg = (
                "Unable to generate nonempty training support after "
                f"{_MAX_NONEMPTY_RESAMPLES} attempts."
            )
            raise RuntimeError(msg)
        return observed
