from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import pytest

from scripts.paper import plot_exact_study as plotting
from scripts.paper.budget_problem import BudgetProblem, Policy
from scripts.paper.budget_study import (
    run_study,
    study_task,
    write_results,
)
from scripts.paper.exact_study import (
    ARM_AGNOSTIC,
    ARM_COMPLETE,
    ARM_GENERATIVE,
    ARM_LOCAL,
    BoolArray,
    Dataset,
    Problem,
    _study_task,
    fit_plans,
)
from scripts.paper.plot_exact_study import render

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def direct_values(  # noqa: C901 - deliberately direct reference recursion
    problem: BudgetProblem, data: Dataset, *, alias: bool
) -> Callable[[int, int, int], float]:
    """Independent reference using individual training instances and recursion."""
    lookup = {int(s): i for i, s in enumerate(problem.states)}

    @cache
    def support(state: int) -> BoolArray:
        keep = np.ones(len(data.y), dtype=bool)
        for a in range(problem.d):
            digit = state // 3**a % 3
            if digit:
                keep &= data.available[:, a] & (data.x[:, a] == digit - 1)
        return keep

    @cache
    def stop(state: int) -> float:
        keep = support(state)
        p1 = (float(data.y[keep].sum()) + 0.5) / (int(keep.sum()) + 1)
        return p1 if problem.prediction[lookup[state]] else 1 - p1

    def actions(state: int) -> list[int]:
        acquired = [a for a in range(problem.d) if state // 3**a % 3]
        spent = sum(int(problem.costs[a]) for a in acquired)
        return [
            a
            for a in range(problem.d)
            if a not in acquired and spent + problem.costs[a] <= problem.budget
        ]

    @cache
    def q(state: int, a: int, horizon: int) -> float:
        selected = support(state) & data.available[:, a]
        children = [state + (value + 1) * 3**a for value in (0, 1)]
        values = [stop(child) for child in children]
        if alias and (state == 0 or horizon > 1):
            targets = []
            for row in np.flatnonzero(selected):
                child = children[int(data.x[row, a])]
                value = stop(child)
                if horizon > 1:
                    for next_action in actions(child):
                        if data.available[row, next_action]:
                            value = max(
                                value, q(child, next_action, horizon - 1)
                            )
                targets.append(value)
            return (sum(targets) + 0.5) / (len(targets) + 1)
        if horizon > 1:
            values = [
                max(
                    [stop(child)]
                    + [q(child, nxt, horizon - 1) for nxt in actions(child)]
                )
                for child in children
            ]
        ones = np.count_nonzero(selected & (data.x[:, a] == 1))
        p1 = (ones + 0.5) / (np.count_nonzero(selected) + 1)
        return float((1 - p1) * values[0] + p1 * values[1])

    return q


@pytest.mark.parametrize("budget", [2, 3, 4])
def test_solver_matches_instance_level_bellman_recursion(budget: int) -> None:
    problem = BudgetProblem(2 * budget, budget)
    data = problem.sample(np.random.default_rng(8), 31, 0.5)
    plans = problem.fit(data)
    complete = Dataset(data.x, data.y, np.ones_like(data.available))
    keep = data.available.all(axis=1)
    filtered = Dataset(data.x[keep], data.y[keep], data.available[keep])
    for arm, view in (
        (ARM_LOCAL, filtered),
        (ARM_GENERATIVE, data),
        (ARM_AGNOSTIC, data),
        (ARM_COMPLETE, complete),
    ):
        reference = direct_values(problem, view, alias=arm == ARM_AGNOSTIC)
        for horizon in (1, budget):
            for parent, action in zip(*np.nonzero(problem.legal), strict=True):
                remaining = budget - int(problem.spent[parent])
                expected = reference(
                    int(problem.states[parent]),
                    int(action),
                    min(horizon, remaining),
                )
                assert plans[arm, horizon].q[parent, action] == pytest.approx(
                    expected, abs=1e-12
                )


@pytest.mark.parametrize(("seed", "n"), [(0, 0), (4, 10), (107, 100)])
def test_budget_two_preserves_original_sampling_values_and_policies(
    seed: int, n: int
) -> None:
    problem, old = BudgetProblem(10, 2), Problem(10)
    data = problem.sample(np.random.default_rng(seed), n, 0.7)
    original = old.sample(np.random.default_rng(seed), n, 0.7)
    for field in ("x", "y", "available"):
        np.testing.assert_array_equal(
            getattr(data, field), getattr(original, field)
        )
    plans = problem.fit(data)
    for key, reference in fit_plans(old, data).items():
        np.testing.assert_allclose(
            plans[key].q[0], reference.root_q, atol=1e-12
        )
        assert plans[key].actions[0] == reference.root_action
        assert problem.evaluate(plans[key]) == old.evaluate(reference)
    rows = study_task((2, 0.7, n, 0, seed))
    assert [row[:-1] for row in rows] == _study_task(
        (10, 0.7, n, 0, seed)
    ).regrets


@pytest.mark.parametrize("budget", [2, 3, 4])
def test_population_policies_and_empty_support(budget: int) -> None:
    problem = BudgetProblem(10, budget)
    plans = problem.model_plans(problem.truth_stop, problem.truth_transition)
    assert problem.evaluate(plans[1]) == 0.75
    assert problem.evaluate(plans[budget]) == 1
    assert plans[1].q[0, 0] == 0.5
    assert plans[budget].q[0, 0] == 1
    assert plans[budget].q[0, 3] == 0.75
    data = problem.sample(np.random.default_rng(0), 0, 0.7)
    for plan in problem.fit(data).values():
        assert plan.actions[0] == -1
        assert problem.evaluate(plan) == 0.5
        assert np.isfinite(plan.q[problem.legal]).all()
    bad_actions = plans[1].actions.copy()
    bad_actions[problem.layers[-1][0]] = 0
    with pytest.raises(ValueError, match="illegal acquisition"):
        problem.evaluate(Policy(bad_actions, plans[1].q))


def test_unavailable_values_do_not_leak_into_incomplete_estimators() -> None:
    problem = BudgetProblem(6, 3)
    data = problem.sample(np.random.default_rng(4), 100, 0.5)
    changed = Dataset(
        np.where(data.available, data.x, 1 - data.x).astype(np.uint8),
        data.y,
        data.available,
    )
    first, second = problem.fit(data), problem.fit(changed)
    for key in first:
        if key[0] != ARM_COMPLETE:
            np.testing.assert_array_equal(first[key].q, second[key].q)


def test_budget_smoke_is_paired_reproducible_and_renders(
    tmp_path: Path,
) -> None:
    rows = run_study(reps=1, seed=0, jobs=1, smoke=True)
    assert rows == run_study(reps=1, seed=0, jobs=2, smoke=True)
    assert len(rows) == 3 * 3 * 3 * 4 * 2
    assert {row.budget for row in rows} == {2, 3, 4}
    path = tmp_path / "budget.csv"
    write_results(path, rows)
    render(path, tmp_path / "budget", vary_budget=True)
    assert (tmp_path / "budget.pdf").stat().st_size > 0


def test_combined_plot_selects_paired_horizons_and_fixed_slices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Include off-slice rates and sweeps so an accidental pool is detectable.
    curves = {
        plotting.Curve(arm, d, p, horizon, budget=budget): plotting.Estimate(
            np.array([10, 100]),
            np.array([d + p, budget + horizon]),
            np.zeros(2),
        )
        for arm, _ in plotting.PANELS
        for d in (6, 8, 10)
        for p in plotting.COLORS
        for budget in (2, 3, 4)
        for horizon in (1, budget)
    }
    monkeypatch.setattr(plotting, "read_means", lambda _: curves)
    figure, axes = plotting._figure()  # noqa: SLF001
    monkeypatch.setattr(plotting, "_figure", lambda: (figure, axes))
    plotting.render_combined(
        tmp_path / "dimension.csv",
        tmp_path / "budget.csv",
        tmp_path / "combined",
    )
    for row_index, row in enumerate(axes):
        values = (6, 8, 10) if row_index == 0 else (2, 3, 4)
        for axis in row:
            assert len(axis.lines) == 6
            for index, line in enumerate(axis.lines):
                value = values[index // 2]
                budget = 2 if row_index == 0 else value
                d = value if row_index == 0 else 10
                myopic = bool(index % 2)
                horizon = 1 if myopic else budget
                np.testing.assert_array_equal(
                    line.get_ydata(),
                    [d + plotting.MAIN_RATE, budget + horizon],
                )
                assert line.get_linestyle() == ("--" if myopic else "-")
                assert line.get_marker() == "None"
                assert line.get_color() == (
                    plotting.MYOPIC_COLOR
                    if myopic
                    else (plotting.DIMENSION_COLORS, plotting.BUDGET_COLORS)[
                        row_index
                    ][index // 2]
                )
    assert (tmp_path / "combined.pdf").stat().st_size > 0
