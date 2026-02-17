# ruff: noqa
from __future__ import annotations

import numpy as np
from sklearn.base import _fit_context

from afabench.missing_values.malearn.base import BaseMADT, RegressorMixin

VARIANCE_THRESHOLD = np.finfo("double").eps


class MADTRegressor(RegressorMixin, BaseMADT):
    """Missingness-avoiding decision tree regressor."""

    criterion = "variance_reduction"

    def __init__(
        self,
        max_depth: int = 3,
        random_state: int | None = None,
        alpha: float = 1.0,
        compute_rho_per_node: bool = False,
        ccp_alpha: float = 0.0,
    ):
        super().__init__(
            max_depth, random_state, alpha, compute_rho_per_node, ccp_alpha
        )

    def _get_node_value(self, y, sample_weight):
        return np.average(y, weights=sample_weight)

    def _get_node_impurity(self, node_value, y, sample_weight):
        del node_value
        return np.var(y * sample_weight)

    def _is_homogeneous(self, node_value, y, sample_weight):
        del node_value
        return np.var(y * sample_weight) < VARIANCE_THRESHOLD

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, M=None, sample_weight=None):
        return super()._fit(X, y, M, sample_weight=sample_weight)

    def predict_proba(self, X, check_input=True):
        return self.predict(X, check_input)

    def _best_split(self, X, y, feature, sample_weight, M=None):
        sorted_indices = np.argsort(X[:, feature])
        X_sorted = X[sorted_indices, feature]
        y_sorted = y[sorted_indices]
        sample_weight_sorted = sample_weight[sorted_indices]

        y_sum_low = 0.0
        y2_sum_low = 0.0

        y_sum_high = np.sum(y_sorted * sample_weight_sorted)
        y2_sum_high = np.sum((y_sorted**2) * sample_weight_sorted)

        n_low = 0.0
        n_high = np.sum(sample_weight)

        max_score = -np.inf
        i_max_score = None

        n = len(y)
        for i in range(n - 1):
            yi = y_sorted[i]
            wi = sample_weight_sorted[i]

            n_low += wi
            y_sum_low += yi * wi
            y2_sum_low += yi**2 * wi

            n_high -= wi
            y_sum_high -= yi * wi
            y2_sum_high -= yi**2 * wi

            if n_low == 0:
                continue

            if n_high == 0:
                break

            if np.isclose(X_sorted[i], X_sorted[i + 1]):
                continue

            score = (y2_sum_low - y_sum_low**2 / n_low) + (
                y2_sum_high - y_sum_high**2 / n_high
            )

            if score > max_score:
                max_score = score
                i_max_score = i

        if i_max_score is None:
            return -np.inf, None, None

        if M is not None:
            max_score -= self.alpha * np.mean(sample_weight * M[:, feature])

        split_threshold = 0.5 * (
            X_sorted[i_max_score] + X_sorted[i_max_score + 1]
        )
        return max_score, feature, split_threshold
