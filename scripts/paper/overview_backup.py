from __future__ import annotations

import numpy as np

from scripts.paper.exact_study import (
    ARM_AGNOSTIC,
    ARM_GENERATIVE,
    ARM_LOCAL,
    BUDGET,
    CONTEXT,
    LAPLACE,
    SHORTCUT,
    CountTables,
    Dataset,
    FixedPredictor,
    FloatArray,
    IntArray,
    Problem,
    _base_plan_quantities,
    _smoothed_probability,
    build_count_tables,
)

ARMS = (ARM_AGNOSTIC, ARM_LOCAL, ARM_GENERATIVE)


def _root_values_model_based(
    tables: CountTables, costs: IntArray, predictor: FixedPredictor
) -> FloatArray:
    quantities = _base_plan_quantities(tables, costs, predictor)
    root_q = np.full(len(costs), -np.inf, dtype=np.float64)
    for action, cost in enumerate(costs):
        if cost > BUDGET:
            continue
        counts = tables.single_count[action]
        p_one = float(_smoothed_probability(counts[1], counts.sum()))
        values = quantities.single_stop[action].copy()
        if cost < BUDGET:
            values = np.maximum(values, quantities.q_last[action].max(axis=1))
        root_q[action] = (1 - p_one) * values[0] + p_one * values[1]
    return root_q


def _root_values_agnostic(
    data: Dataset,
    tables: CountTables,
    costs: IntArray,
    predictor: FixedPredictor,
) -> FloatArray:
    """Return the root action-values behind plan_mask_agnostic."""
    quantities = _base_plan_quantities(tables, costs, predictor)
    root_q = np.full(len(costs), -np.inf, dtype=np.float64)
    for action, cost in enumerate(costs):
        if cost > BUDGET:
            continue
        selected = data.available[:, action]
        count = int(selected.sum())
        if count == 0:
            root_q[action] = 0.5
            continue
        values = quantities.single_stop[action, data.x[:, action]].copy()
        for continuation, continuation_cost in enumerate(costs):
            if cost + continuation_cost > BUDGET:
                continue
            candidate = quantities.q_last[
                action, data.x[:, action], continuation
            ]
            values = np.where(
                data.available[:, continuation],
                np.maximum(values, candidate),
                values,
            )
        root_q[action] = (float(values[selected].sum()) + LAPLACE) / (
            count + 2 * LAPLACE
        )
    return root_q


def draws(
    d: int, p_miss: float, n: int, reps: int, seed: int
) -> dict[str, FloatArray]:
    """One (context, shortcut) action-value pair per approach per replication."""
    problem = Problem(d)
    predictor = problem.fixed_predictor()
    out = {arm: np.empty((reps, 2)) for arm in ARMS}
    for rep in range(reps):
        rng = np.random.default_rng(
            np.random.SeedSequence((seed, d, round(1000 * p_miss), n, rep))
        )
        data = problem.sample(rng, n, p_miss)
        pooled = build_count_tables(data)
        complete_only = build_count_tables(data, complete_only=True)
        values = {
            ARM_AGNOSTIC: _root_values_agnostic(
                data, pooled, problem.costs, predictor
            ),
            ARM_LOCAL: _root_values_model_based(
                complete_only, problem.costs, predictor
            ),
            ARM_GENERATIVE: _root_values_model_based(
                pooled, problem.costs, predictor
            ),
        }
        for arm, root_q in values.items():
            out[arm][rep] = (root_q[CONTEXT], root_q[SHORTCUT])
    return out
