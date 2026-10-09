"""
Task sampler for L2M's task prior (paper Appendix A.5.2, A.5.3).

Both l2m_real_feature_prior and l2m_synthetic_feature_prior draw every
training task from `sample_task`; the feature source is their only
difference. Independently reimplemented from the paper, not from the
authors' unlicensed repository.

Missingness is MCAR: each feature's missing rate is drawn independently of
the features themselves, per the maintainer's decision overriding the
paper's MAR mechanism for MiniBooNE (which the paper only defines in terms
of baseline covariates MiniBooNE does not have).

The paper leaves several task-prior parameters unstated; each choice made
here is documented at the point it is used.
"""

from dataclasses import dataclass
from enum import StrEnum

import torch

from afabench.components.methods.discriminative.l2m.models import (
    TaskFeatures,
    TaskLabels,
    TaskMask,
)
from afabench.core.types import Features


class FeatureSource(StrEnum):
    """Where a task's features come from; config values are the names."""

    real = "real"
    synthetic = "synthetic"


# The uniform box is this benchmark's own choice for the synthetic feature
# source; the paper does not describe it.
_SYNTHETIC_FEATURE_LOW = -2.0
_SYNTHETIC_FEATURE_HIGH = 2.0

# BNN labelling-function prior (paper A.5.2): hidden width is stated.
_BNN_HIDDEN_DIM = 8
# Every informative-feature subset size from a single feature to every
# feature is allowed, so both sparse and dense labelling functions occur.
_MIN_INFORMATIVE_FEATURES = 1
# Paper: "1-3 clusters" (the "11-33" HTML rendering is an artifact).
_N_CLUSTERS_LOW = 1
_N_CLUSTERS_HIGH = 3
# Per-input-feature importance weighting and per-layer weight scaling,
# both unspecified ranges in the paper; centered at 1 so an "average" draw
# reproduces a plain random network.
_IMPORTANCE_WEIGHT_RANGE = (0.5, 1.5)
_LAYER_SCALE_RANGE = (0.5, 2.0)
# Logit temperature range, unspecified in the paper.
_TEMPERATURE_RANGE = (0.5, 2.0)
# Binary target prevalence, stated explicitly in the paper.
_PREVALENCE_RANGE = (0.05, 0.95)
# Bisection search for the bias that matches the mean label prevalence;
# the paper does not say how the match is computed.
_PREVALENCE_BIAS_RADIUS = 20.0
_PREVALENCE_BISECTION_STEPS = 30


@dataclass(frozen=True, kw_only=True)
class Task:
    """One task drawn from the task prior: a sequence of instances."""

    features: TaskFeatures
    labels: TaskLabels
    feature_mask: TaskMask


def sample_task(
    feature_source: FeatureSource,
    *,
    n_features: int,
    sequence_length: int,
    label_shape: torch.Size,
    missingness_cap: float = 0.5,
    feature_pool: Features | None = None,
    seed: int,
) -> Task:
    """
    Draw one reproducible task from the task prior (paper A.5.2, A.5.3).

    `feature_source` selects instances of the real pool
    (`l2m_real_feature_prior`) or the uniform box on [-2, 2]
    (`l2m_synthetic_feature_prior`). Real-pool tasks draw instances
    without replacement within the task. `missingness_cap`
    bounds the per-feature MCAR missing rate (maintainer decision; see the
    module docstring), defaulting to the paper's 0.5. A cap of 0 leaves
    every feature observed.
    """
    if sequence_length < 2:
        msg = (
            f"sequence_length={sequence_length} must be at least 2, "
            "for one context instance and one query"
        )
        raise ValueError(msg)
    if not 0.0 <= missingness_cap <= 1.0:
        msg = f"missingness_cap={missingness_cap} must be in [0, 1]"
        raise ValueError(msg)
    if len(label_shape) != 1 or label_shape[0] < 2:
        msg = f"label_shape={tuple(label_shape)} must be (n_classes,) with n_classes >= 2"
        raise ValueError(msg)

    generator = torch.Generator().manual_seed(seed)
    features = _sample_features(
        feature_source, feature_pool, n_features, sequence_length, generator
    )
    labels = _sample_bnn_labels(features, int(label_shape[0]), generator)
    feature_mask = _sample_mcar_mask(
        sequence_length, n_features, missingness_cap, generator
    )
    return Task(features=features, labels=labels, feature_mask=feature_mask)


def _sample_features(
    feature_source: FeatureSource,
    feature_pool: Features | None,
    n_features: int,
    sequence_length: int,
    generator: torch.Generator,
) -> TaskFeatures:
    match feature_source:
        case FeatureSource.synthetic:
            if feature_pool is not None:
                msg = (
                    "feature_pool must be None for the synthetic feature "
                    "source"
                )
                raise ValueError(msg)
            unit = torch.rand(sequence_length, n_features, generator=generator)
            return (
                _SYNTHETIC_FEATURE_LOW
                + (_SYNTHETIC_FEATURE_HIGH - _SYNTHETIC_FEATURE_LOW) * unit
            )
        case FeatureSource.real:
            if feature_pool is None:
                msg = "feature_pool is required for the real feature source"
                raise ValueError(msg)
            if feature_pool.ndim != 2 or feature_pool.shape[1] != n_features:
                msg = (
                    f"feature_pool shape {tuple(feature_pool.shape)} must be "
                    f"(pool_size, {n_features})"
                )
                raise ValueError(msg)
            if feature_pool.shape[0] < sequence_length:
                msg = (
                    f"feature_pool has {feature_pool.shape[0]} instances, "
                    f"fewer than sequence_length={sequence_length}; a task "
                    "draws instances without replacement"
                )
                raise ValueError(msg)
            indices = torch.randperm(
                feature_pool.shape[0], generator=generator
            )
            return feature_pool[indices[:sequence_length]].clone()


def _sample_bnn_labels(
    features: TaskFeatures, n_classes: int, generator: torch.Generator
) -> TaskLabels:
    n_features = features.shape[1]

    n_clusters = int(
        torch.randint(
            _N_CLUSTERS_LOW,
            _N_CLUSTERS_HIGH + 1,
            (1,),
            generator=generator,
        ).item()
    )
    # Cluster centers are drawn around the task's own feature statistics,
    # so the Gaussian is well-scaled whether features come from the real
    # pool or the synthetic box (paper does not parameterize the Gaussian).
    # A constant feature puts every center at its value, so it does not
    # separate clusters; if every feature is constant, all instances fall
    # in the first cluster.
    centers = features.mean(dim=0) + features.std(dim=0) * torch.randn(
        n_clusters, n_features, generator=generator
    )
    cluster_ids = torch.cdist(features, centers).argmin(dim=-1)

    informative_mask = torch.zeros(n_clusters, n_features, dtype=torch.bool)
    for cluster in range(n_clusters):
        subset_size = int(
            torch.randint(
                _MIN_INFORMATIVE_FEATURES,
                n_features + 1,
                (1,),
                generator=generator,
            ).item()
        )
        subset = torch.randperm(n_features, generator=generator)[:subset_size]
        informative_mask[cluster, subset] = True
    masked_input = features * informative_mask[cluster_ids]

    importance_weights = _uniform(
        (n_features,), _IMPORTANCE_WEIGHT_RANGE, generator
    )
    masked_input = masked_input * importance_weights

    scale1 = _uniform((), _LAYER_SCALE_RANGE, generator)
    w1 = torch.randn(n_features, _BNN_HIDDEN_DIM, generator=generator)
    w1 = w1 * scale1 / (n_features**0.5)
    b1 = torch.randn(_BNN_HIDDEN_DIM, generator=generator) * 0.1
    hidden = torch.tanh(masked_input @ w1 + b1)

    scale2 = _uniform((), _LAYER_SCALE_RANGE, generator)
    w2 = torch.randn(_BNN_HIDDEN_DIM, n_classes, generator=generator)
    w2 = w2 * scale2 / (_BNN_HIDDEN_DIM**0.5)
    b2 = torch.randn(n_classes, generator=generator) * 0.1
    logits = hidden @ w2 + b2

    temperature = _uniform((), _TEMPERATURE_RANGE, generator)
    logits = logits * temperature

    if n_classes == 2:
        prevalence = _uniform((), _PREVALENCE_RANGE, generator)
        logits = _shift_to_match_prevalence(logits, prevalence)

    probabilities = torch.softmax(logits, dim=-1)
    label_indices = torch.multinomial(
        probabilities, 1, generator=generator
    ).squeeze(-1)
    return torch.nn.functional.one_hot(
        label_indices, num_classes=n_classes
    ).to(features.dtype)


def _uniform(
    shape: tuple[int, ...],
    bounds: tuple[float, float],
    generator: torch.Generator,
) -> torch.Tensor:
    low, high = bounds
    return low + (high - low) * torch.rand(shape, generator=generator)


def _shift_to_match_prevalence(
    logits: torch.Tensor, prevalence: torch.Tensor
) -> torch.Tensor:
    """Bisect for the bias on logit 1 whose mean sigmoid hits `prevalence`."""
    diff = logits[:, 1] - logits[:, 0]
    low = torch.full((), -_PREVALENCE_BIAS_RADIUS)
    high = torch.full((), _PREVALENCE_BIAS_RADIUS)
    for _ in range(_PREVALENCE_BISECTION_STEPS):
        mid = (low + high) / 2
        mean_probability = torch.sigmoid(diff + mid).mean()
        low, high = torch.where(
            mean_probability < prevalence,
            torch.stack((mid, high)),
            torch.stack((low, mid)),
        )
    bias = (low + high) / 2
    return torch.stack((logits[:, 0], logits[:, 1] + bias), dim=-1)


def _sample_mcar_mask(
    sequence_length: int,
    n_features: int,
    missingness_cap: float,
    generator: torch.Generator,
) -> TaskMask:
    # Each feature's own missing rate is drawn once per task, uniformly
    # below the cap; the paper states only the cap, not how per-feature
    # rates below it are drawn.
    missing_rates = missingness_cap * torch.rand(
        n_features, generator=generator
    )
    draws = torch.rand(sequence_length, n_features, generator=generator)
    missing = draws < missing_rates
    return ~missing
