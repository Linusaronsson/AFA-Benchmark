"""
Exact subset-state control for the budget-scaled shortcut problem.

A state is a base-three code: 0 means unacquired, 1/2 mean observed 0/1.
Training histograms use the same codes, with 0 meaning unavailable. Marginal
counts sum over all training patterns extending a queried acquisition state.
"""

from __future__ import annotations

from typing import NamedTuple, final

import numpy as np

from scripts.paper.exact_study import (
    ARM_AGNOSTIC,
    ARM_COMPLETE,
    ARM_GENERATIVE,
    ARM_LOCAL,
    LAPLACE,
    SHORTCUT,
    SHORTCUT_ACCURACY,
    Dataset,
    FloatArray,
    IntArray,
    UIntArray,
)


class Policy(NamedTuple):
    actions: IntArray
    q: FloatArray


def marginal_counts(histogram: FloatArray, d: int) -> FloatArray:
    """Sum support for each acquisition state, retaining observed values."""
    table = histogram.reshape((3,) * d, order="F").copy()
    for axis in range(d):
        view = np.moveaxis(table, axis, 0)
        view[0] = view.sum(axis=0)
    return table.ravel(order="F")


def availability_counts(histogram: FloatArray, d: int) -> FloatArray:
    """Index constraints as base four: any, observed 0/1, or unavailable."""
    table = histogram.reshape((3,) * d, order="F")
    for axis in range(d):
        view = np.moveaxis(table, axis, 0)
        table = np.moveaxis(
            np.stack((view.sum(axis=0), view[1], view[2], view[0])), 0, axis
        )
    return table.ravel(order="F")


@final
class BudgetProblem:
    """A context selects the parity of one of two branches of b-1 bits."""

    def __init__(self, d: int = 10, budget: int = 2):
        if budget not in (2, 3, 4) or not 2 * budget <= d <= 10:
            message = "Require budget in {2,3,4} and 2*budget <= d <= 10."
            raise ValueError(message)
        self.d, self.budget = d, budget
        self.branches = (
            (1, *range(4, budget + 2)),
            (2, *range(budget + 2, 2 * budget)),
        )
        self.costs = np.ones(d, dtype=np.int64)
        self.costs[SHORTCUT] = budget
        self.powers = 3 ** np.arange(d, dtype=np.int64)
        self.mask_powers = 4 ** np.arange(d, dtype=np.int64)
        codes = np.arange(3**d, dtype=np.int64)
        digits = (codes[:, None] // self.powers) % 3
        costs = (digits != 0) @ self.costs
        self.states = codes[costs <= budget]
        self.digits = digits[self.states]
        self.spent = costs[self.states]
        self.layers = [
            np.flatnonzero(self.spent == c) for c in range(budget + 1)
        ]
        self.constraint_codes = self.digits @ self.mask_powers
        lookup = np.full(3**d, -1, dtype=np.int64)
        lookup[self.states] = np.arange(len(self.states))
        self.legal = (self.digits == 0) & (
            self.spent[:, None] + self.costs <= budget
        )
        parents, actions = np.nonzero(self.legal)
        self.children = np.zeros((len(self.states), d, 2), dtype=np.int64)
        for value in (0, 1):
            self.children[parents, actions, value] = lookup[
                self.states[parents] + (value + 1) * self.powers[actions]
            ]

        # Enumerate the true joint distribution, including irrelevant noise.
        x = ((np.arange(2**d)[:, None] >> np.arange(d)) & 1).astype(np.uint8)
        y = self.labels(x)
        weights = np.where(
            x[:, SHORTCUT] == y, SHORTCUT_ACCURACY, 1 - SHORTCUT_ACCURACY
        ) / 2 ** (d - 1)
        codes = (x + 1) @ self.powers
        mass = np.bincount(codes, weights=weights, minlength=3**d).astype(
            np.float64
        )
        positive = np.bincount(
            codes, weights=weights * y, minlength=3**d
        ).astype(np.float64)
        self.truth_count = marginal_counts(mass, d)[self.states]
        self.truth_y1 = marginal_counts(positive, d)[self.states]
        self.prediction = self.truth_y1 / self.truth_count >= 0.5
        self.truth_stop = (
            np.where(
                self.prediction,
                self.truth_y1,
                self.truth_count - self.truth_y1,
            )
            / self.truth_count
        )
        self.truth_transition = self.transitions(self.truth_count, laplace=0)

    def labels(self, x: UIntArray) -> UIntArray:
        parities = [
            np.bitwise_xor.reduce(x[:, branch], axis=1)
            for branch in self.branches
        ]
        return np.where(x[:, 0] == 0, parities[0], parities[1]).astype(
            np.uint8
        )

    def sample(
        self, rng: np.random.Generator, n: int, p_miss: float
    ) -> Dataset:
        if n < 0 or not 0 <= p_miss < 1:
            message = "Require n >= 0 and 0 <= p_miss < 1."
            raise ValueError(message)
        x = rng.integers(0, 2, size=(n, self.d), dtype=np.uint8)
        y = self.labels(x)
        faithful = rng.random(n) < SHORTCUT_ACCURACY
        x[:, SHORTCUT] = np.where(faithful, y, 1 - y)
        return Dataset(x, y, rng.random((n, self.d)) >= p_miss)

    def histograms(
        self, data: Dataset, *, complete: bool = False
    ) -> tuple[FloatArray, FloatArray]:
        digits = data.x + 1 if complete else (data.x + 1) * data.available
        codes = digits @ self.powers
        return (
            np.bincount(codes, minlength=3**self.d).astype(np.float64),
            np.bincount(codes, weights=data.y, minlength=3**self.d).astype(
                np.float64
            ),
        )

    def transitions(
        self, count: FloatArray, *, laplace: float = LAPLACE
    ) -> FloatArray:
        children = count[self.children]
        return (children[:, :, 1] + laplace) / (
            children.sum(axis=2) + 2 * laplace
        )

    def policy(self, q: FloatArray, stop: FloatArray) -> Policy:
        best = stop.copy()
        actions = np.full(len(stop), -1, dtype=np.int64)
        for action in range(self.d):
            improve = q[:, action] > best + 1e-12
            best[improve] = q[improve, action]
            actions[improve] = action
        return Policy(actions, q)

    def model_plans(
        self, stop: FloatArray, transition: FloatArray
    ) -> dict[int, Policy]:
        after = stop[self.children]
        myopic = (1 - transition) * after[:, :, 0] + transition * after[
            :, :, 1
        ]
        myopic[~self.legal] = -np.inf
        full = myopic.copy()
        value = stop.copy()
        for layer in reversed(self.layers):
            after = value[self.children[layer]]
            full[layer] = np.where(
                self.legal[layer],
                (1 - transition[layer]) * after[:, :, 0]
                + transition[layer] * after[:, :, 1],
                -np.inf,
            )
            value[layer] = np.maximum(stop[layer], full[layer].max(axis=1))
        return {
            1: self.policy(myopic, stop),
            self.budget: self.policy(full, stop),
        }

    def masked_backup(
        self,
        q: FloatArray,
        stop: FloatArray,
        constraints: FloatArray,
        layer: IntArray,
    ) -> FloatArray:
        """
        Integrate max over the empirical legal sets without enumerating rows.

        In descending action-value order, each term counts instances where
        that action is available and every better action is unavailable.
        """
        codes = self.constraint_codes[layer].copy()
        total = constraints[codes] * stop[layer]
        order = np.argsort(-q[layer], axis=1, kind="stable")
        for rank in range(self.d):
            action = order[:, rank]
            candidate = q[layer, action]
            keep = candidate > stop[layer]
            missing = codes[keep] + 3 * self.mask_powers[action[keep]]
            count = constraints[codes[keep]] - constraints[missing]
            total[keep] += count * (candidate[keep] - stop[layer[keep]])
            codes[keep] = missing
        return total

    def alias_plans(
        self,
        stop: FloatArray,
        count: FloatArray,
        transition: FloatArray,
        constraints: FloatArray,
    ) -> dict[int, Policy]:
        plans = self.model_plans(stop, transition)
        myopic = plans[1].q.copy()
        full = myopic.copy()
        backup = stop * count
        # Match the original b=2 estimator: plug-in one-step leaf values,
        # mask-averaged multi-step targets, and smoothed empirical root targets.
        root_children = self.children[0]
        myopic[0] = (backup[root_children].sum(axis=1) + LAPLACE) / (
            count[root_children].sum(axis=1) + 2 * LAPLACE
        )
        myopic[0, ~self.legal[0]] = -np.inf
        for layer in reversed(self.layers):
            if self.spent[layer[0]] < self.budget - 1:
                full[layer] = np.where(
                    self.legal[layer],
                    (backup[self.children[layer]].sum(axis=2) + LAPLACE)
                    / (count[self.children[layer]].sum(axis=2) + 2 * LAPLACE),
                    -np.inf,
                )
            backup[layer] = self.masked_backup(full, stop, constraints, layer)
        return {
            1: self.policy(myopic, stop),
            self.budget: self.policy(full, stop),
        }

    def fit(self, data: Dataset) -> dict[tuple[str, int], Policy]:
        pooled, pooled_y1 = self.histograms(data)
        complete, complete_y1 = self.histograms(data, complete=True)
        # Histogram filtering uses all d features, not just legal query states.
        codes = np.arange(3**self.d)
        all_observed = np.all((codes[:, None] // self.powers) % 3 != 0, axis=1)
        histograms = {
            ARM_LOCAL: (pooled * all_observed, pooled_y1 * all_observed),
            ARM_GENERATIVE: (pooled, pooled_y1),
            ARM_COMPLETE: (complete, complete_y1),
        }
        fitted = {}
        for arm, (hist, positive) in histograms.items():
            count = marginal_counts(hist, self.d)[self.states]
            y1 = marginal_counts(positive, self.d)[self.states]
            p1 = (y1 + LAPLACE) / (count + 2 * LAPLACE)
            stop = np.where(self.prediction, p1, 1 - p1)
            transition = self.transitions(count)
            fitted.update(
                {
                    (arm, h): plan
                    for h, plan in self.model_plans(stop, transition).items()
                }
            )
            if arm == ARM_GENERATIVE:
                aliases = self.alias_plans(
                    stop,
                    count,
                    transition,
                    availability_counts(pooled, self.d),
                )
                fitted.update(
                    {(ARM_AGNOSTIC, h): plan for h, plan in aliases.items()}
                )
        return fitted

    def evaluate(self, policy: Policy) -> float:
        if np.any((policy.actions < -1) | (policy.actions >= self.d)):
            message = "Invalid policy action."
            raise ValueError(message)
        acquired = np.flatnonzero(policy.actions >= 0)
        if not self.legal[acquired, policy.actions[acquired]].all():
            message = "Policy contains an illegal acquisition."
            raise ValueError(message)
        value = self.truth_stop.copy()
        for layer in reversed(self.layers):
            selected = layer[policy.actions[layer] >= 0]
            action = policy.actions[selected]
            children = self.children[selected, action]
            p1 = self.truth_transition[selected, action]
            value[selected] = (1 - p1) * value[children[:, 0]] + p1 * value[
                children[:, 1]
            ]
        return float(value[0])
