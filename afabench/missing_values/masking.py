"""
Missingness mask generators used by AFA warm-start initializers.

This module adapts the mask simulation functions from:
https://github.com/BorisMuzellec/MissingDataOT

The adaptation is based on the implementation used in MA-learn:
https://github.com/antmats/malearn
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import optimize

__all__ = [
    "MAR_mask",
    "MNAR_mask_logistic",
    "MNAR_mask_quantiles",
    "MNAR_self_mask_logistic",
]


def _get_random_state(seed: int | None) -> np.random.RandomState:
    return np.random.RandomState(seed)


def _get_torch_generator(
    seed: int | None, device: torch.device
) -> torch.Generator:
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def _to_tensor(
    x: torch.Tensor | np.ndarray,
) -> tuple[torch.Tensor, bool]:
    is_torch = torch.is_tensor(x)
    if is_torch:
        return x, True
    return torch.from_numpy(x), False


def _to_original_type(
    tensor: torch.Tensor, as_torch: bool
) -> torch.BoolTensor | np.ndarray:
    if as_torch:
        return tensor.bool()
    return tensor.cpu().numpy().astype(bool)


def _quantile(
    x: torch.Tensor, q: float, dim: int | None = None
) -> torch.Tensor:
    if dim is None:
        flat = x.reshape(-1)
        k = int(q * flat.shape[0])
        k = min(max(k, 1), flat.shape[0])
        return flat.kthvalue(k).values
    k = int(q * x.shape[dim])
    k = min(max(k, 1), x.shape[dim])
    return x.kthvalue(k, dim=dim).values


def _pick_coeffs(
    x: torch.Tensor,
    idxs_obs: np.ndarray | None = None,
    idxs_nas: np.ndarray | None = None,
    *,
    self_mask: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    eps = torch.finfo(x.dtype).eps
    _, d = x.shape
    if self_mask:
        coeffs = torch.randn(
            d,
            generator=generator,
            device=x.device,
            dtype=x.dtype,
        )
        Wx = x * coeffs
        std = torch.std(Wx, 0, unbiased=False)
        std = torch.nan_to_num(std, nan=1.0).clamp_min(eps)
        coeffs /= std
        return coeffs

    assert idxs_obs is not None
    assert idxs_nas is not None
    d_obs = len(idxs_obs)
    d_na = len(idxs_nas)
    coeffs = torch.randn(
        d_obs,
        d_na,
        generator=generator,
        device=x.device,
        dtype=x.dtype,
    )
    Wx = x[:, idxs_obs].mm(coeffs)
    std = torch.std(Wx, 0, keepdim=True, unbiased=False)
    std = torch.nan_to_num(std, nan=1.0).clamp_min(eps)
    coeffs /= std
    return coeffs


def _fit_intercepts(
    features: torch.Tensor,
    coeffs: torch.Tensor,
    p: float,
    *,
    self_mask: bool = False,
) -> torch.Tensor:
    eps = torch.finfo(features.dtype).eps
    p_clamped = float(np.clip(p, eps, 1.0 - eps))
    target_logit = float(np.log(p_clamped / (1.0 - p_clamped)))

    def _solve_intercept(logits: torch.Tensor) -> float:
        # Single-sample case has a closed-form solution.
        if logits.numel() == 1:
            return target_logit - float(logits.item())

        mean_logit = float(logits.mean().item())
        width = 50.0

        def f(intercept: float) -> float:
            return torch.sigmoid(logits + intercept).mean().item() - p_clamped

        # Adapt bracket around the empirical center.
        for _ in range(8):
            a = target_logit - mean_logit - width
            b = target_logit - mean_logit + width
            fa = f(a)
            fb = f(b)
            if np.isfinite(fa) and np.isfinite(fb) and fa * fb <= 0:
                return float(optimize.bisect(f, a, b))
            width *= 2.0

        # Fallback: use endpoint closest to target prevalence.
        a = target_logit - mean_logit - width
        b = target_logit - mean_logit + width
        fa = f(a)
        fb = f(b)
        return float(a if abs(fa) <= abs(fb) else b)

    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(
            d, dtype=features.dtype, device=features.device
        )
        for idx in range(d):
            logits = features[:, idx] * coeffs[idx]
            intercepts[idx] = _solve_intercept(logits)
        return intercepts

    d_obs, d_na = coeffs.shape
    intercepts = torch.zeros(
        d_na, dtype=features.dtype, device=features.device
    )
    for idx in range(d_na):
        logits = features.mv(coeffs[:, idx])
        intercepts[idx] = _solve_intercept(logits)
    return intercepts


def MAR_mask(  # noqa: N802
    x: torch.Tensor | np.ndarray,
    p: float,
    p_obs: float,
    seed: int | None = None,
) -> torch.BoolTensor | np.ndarray:
    random_state = _get_random_state(seed)
    tensor_x, as_torch = _to_tensor(x)
    generator = _get_torch_generator(seed, tensor_x.device)

    n, d = tensor_x.shape
    mask = torch.zeros(n, d, dtype=torch.bool, device=tensor_x.device)

    d_obs = max(int(p_obs * d), 1)
    idxs_obs = random_state.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    coeffs = _pick_coeffs(
        tensor_x, idxs_obs=idxs_obs, idxs_nas=idxs_nas, generator=generator
    )
    intercepts = _fit_intercepts(tensor_x[:, idxs_obs], coeffs, p)
    ps = torch.sigmoid(tensor_x[:, idxs_obs].mm(coeffs) + intercepts)
    bernoulli = torch.rand(
        n,
        len(idxs_nas),
        generator=generator,
        device=tensor_x.device,
        dtype=tensor_x.dtype,
    )
    mask[:, idxs_nas] = bernoulli < ps
    return _to_original_type(mask, as_torch)


def MNAR_mask_logistic(  # noqa: N802
    x: torch.Tensor | np.ndarray,
    p: float,
    p_params: float = 0.3,
    *,
    exclude_inputs: bool = True,
    seed: int | None = None,
) -> torch.BoolTensor | np.ndarray:
    random_state = _get_random_state(seed)
    tensor_x, as_torch = _to_tensor(x)
    generator = _get_torch_generator(seed, tensor_x.device)

    n, d = tensor_x.shape
    mask = torch.zeros(n, d, dtype=torch.bool, device=tensor_x.device)

    if exclude_inputs:
        d_params = max(int(p_params * d), 1)
        idxs_params = random_state.choice(d, d_params, replace=False)
        idxs_nas = np.array([i for i in range(d) if i not in idxs_params])
    else:
        idxs_params = np.arange(d)
        idxs_nas = np.arange(d)

    coeffs = _pick_coeffs(
        tensor_x,
        idxs_obs=idxs_params,
        idxs_nas=idxs_nas,
        generator=generator,
    )
    intercepts = _fit_intercepts(tensor_x[:, idxs_params], coeffs, p)
    ps = torch.sigmoid(tensor_x[:, idxs_params].mm(coeffs) + intercepts)

    bernoulli = torch.rand(
        n,
        len(idxs_nas),
        generator=generator,
        device=tensor_x.device,
        dtype=tensor_x.dtype,
    )
    mask[:, idxs_nas] = bernoulli < ps

    if exclude_inputs:
        mask[:, idxs_params] = (
            torch.rand(
                n,
                len(idxs_params),
                generator=generator,
                device=tensor_x.device,
                dtype=tensor_x.dtype,
            )
            < p
        )

    return _to_original_type(mask, as_torch)


def MNAR_self_mask_logistic(  # noqa: N802
    x: torch.Tensor | np.ndarray, p: float, seed: int | None = None
) -> torch.BoolTensor | np.ndarray:
    tensor_x, as_torch = _to_tensor(x)
    generator = _get_torch_generator(seed, tensor_x.device)

    n, d = tensor_x.shape
    coeffs = _pick_coeffs(tensor_x, self_mask=True, generator=generator)
    intercepts = _fit_intercepts(tensor_x, coeffs, p, self_mask=True)
    ps = torch.sigmoid(tensor_x * coeffs + intercepts)

    bernoulli = torch.rand(
        n,
        d,
        generator=generator,
        device=tensor_x.device,
        dtype=tensor_x.dtype,
    )
    mask = bernoulli < ps
    return _to_original_type(mask, as_torch)


def MNAR_mask_quantiles(  # noqa: N802
    x: torch.Tensor | np.ndarray,
    p: float,
    q: float,
    p_params: float,
    *,
    cut: str = "both",
    MCAR: bool = False,  # noqa: N803
    seed: int | None = None,
) -> torch.BoolTensor | np.ndarray:
    random_state = _get_random_state(seed)
    tensor_x, as_torch = _to_tensor(x)
    generator = _get_torch_generator(seed, tensor_x.device)

    n, d = tensor_x.shape
    mask = torch.zeros(n, d, dtype=torch.bool, device=tensor_x.device)

    d_na = max(int(p_params * d), 1)
    idxs_na = random_state.choice(d, d_na, replace=False)

    if cut == "upper":
        quants = _quantile(tensor_x[:, idxs_na], 1 - q, dim=0)
        extreme_mask = tensor_x[:, idxs_na] >= quants
    elif cut == "lower":
        quants = _quantile(tensor_x[:, idxs_na], q, dim=0)
        extreme_mask = tensor_x[:, idxs_na] <= quants
    elif cut == "both":
        upper_quants = _quantile(tensor_x[:, idxs_na], 1 - q, dim=0)
        lower_quants = _quantile(tensor_x[:, idxs_na], q, dim=0)
        extreme_mask = (tensor_x[:, idxs_na] <= lower_quants) | (
            tensor_x[:, idxs_na] >= upper_quants
        )
    else:
        msg = f"Unknown cut value: {cut}. Expected one of upper/lower/both."
        raise ValueError(msg)

    bernoulli = torch.rand(
        n,
        d_na,
        generator=generator,
        device=tensor_x.device,
        dtype=tensor_x.dtype,
    )
    mask[:, idxs_na] = (bernoulli < p) & extreme_mask

    if MCAR:
        mask = mask | (
            torch.rand(
                n,
                d,
                generator=generator,
                device=tensor_x.device,
                dtype=tensor_x.dtype,
            )
            < p
        )

    return _to_original_type(mask, as_torch)
