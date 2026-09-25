from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np
import pytest

from scripts.paper import exact_study
from scripts.paper.conceptual_constants import AVAILABILITY, render
from scripts.paper.exact_study import (
    ARM_AGNOSTIC,
    ARM_GENERATIVE,
    ARM_LOCAL,
    ARMS,
    BUDGET,
    CONTEXT,
    SHORTCUT,
    Dataset,
    Problem,
    build_count_tables,
    fit_plans,
    plan_mask_agnostic,
    plan_model_based,
    run_study,
    write_action_values,
    write_results,
)
from scripts.paper.plot_exact_study import (
    PANELS,
    Curve,
    effective_size,
    read_means,
)
from scripts.paper.plot_exact_study import (
    render as render_regrets,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_shortcut_problem_has_the_stated_evaluation_values() -> None:
    problem = Problem(6)

    assert problem.costs[SHORTCUT] == BUDGET
    assert problem.optimal_value() == 1.0
    assert problem.label_probability((SHORTCUT,), (1,)) == 0.75


def test_mask_local_counts_only_complete_instances() -> None:
    x = np.zeros((3, 4), dtype=np.uint8)
    y = np.zeros(3, dtype=np.uint8)
    available = np.asarray(
        [
            [True, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
        ]
    )

    tables = build_count_tables(Dataset(x, y, available), complete_only=True)

    assert tables.root_count == 1


def test_all_training_views_share_the_fixed_bayes_predictor() -> None:
    problem = Problem(4)
    x = np.zeros((3, 4), dtype=np.uint8)
    y = np.zeros(3, dtype=np.uint8)
    available = np.asarray(
        [
            [True, True, True, True],
            [True, False, True, True],
            [True, True, False, True],
        ]
    )
    data = Dataset(x, y, available)
    pooled = build_count_tables(data)
    complete = build_count_tables(data, complete_only=True)
    predictor = problem.fixed_predictor()

    plans = (
        plan_model_based(complete, problem.costs, predictor),
        plan_mask_agnostic(data, pooled, problem.costs, predictor),
        plan_model_based(pooled, problem.costs, predictor),
    )

    assert predictor.root == 1
    assert y.mean() == 0
    for plan in plans:
        assert plan.root_prediction == predictor.root
        assert np.array_equal(plan.single_prediction, predictor.single)
        assert np.array_equal(plan.pair_prediction, predictor.pair)


def test_smoke_study_contains_all_theoretical_arms() -> None:
    results = run_study(reps=1, seed=0, jobs=1, smoke=True).regrets

    assert {result.arm for result in results} >= {
        ARM_LOCAL,
        ARM_AGNOSTIC,
        ARM_GENERATIVE,
    }
    assert all(0 <= result.regret <= 0.5 for result in results)


@pytest.mark.parametrize("p_tenths", [3, 5, 7])
def test_population_values_and_myopic_policy(
    monkeypatch: pytest.MonkeyPatch,
    p_tenths: int,
) -> None:
    # Enumerate the exact joint law of X,Y,M as integer multiplicities.
    # Removing smoothing makes these population expectations, not a sample.
    monkeypatch.setattr(exact_study, "LAPLACE", 0.0)
    problem = Problem(4)
    core = np.repeat(problem.core, (32 * problem.weights).astype(int), axis=0)
    masks = problem.core.astype(bool)
    multiplicity = np.prod(np.where(masks, 10 - p_tenths, p_tenths), axis=1)
    available = np.repeat(masks, multiplicity, axis=0)
    x = np.repeat(core, len(available), axis=0)
    y = np.where(x[:, 0] == 0, x[:, 1], x[:, 2]).astype(np.uint8)
    data = Dataset(x, y, np.tile(available, (len(core), 1)))
    # Inconsistent values for a feature paired with itself have zero mass.
    # These undefined diagonal cells are never legal acquisitions.
    with np.errstate(invalid="ignore"):
        plans = fit_plans(problem, data)

    for arm in ARMS:
        myopic, two_step = plans[arm, 1], plans[arm, 2]
        np.testing.assert_allclose(myopic.root_q, [0.5, 0.75, 0.75, 0.75])
        target = 1 - p_tenths / 20 if arm == ARM_AGNOSTIC else 1
        assert two_step.root_q[CONTEXT] == pytest.approx(target)
        assert two_step.root_q[SHORTCUT] == pytest.approx(0.75)
        assert problem.evaluate(myopic) == pytest.approx(0.75)
        assert problem.optimal_value(1) == pytest.approx(0.75)
        # Even the myopic plan retains its last-step acquisition table.
        assert tuple(myopic.single_action[CONTEXT]) == (1, 2)


@pytest.mark.parametrize("horizon", [1, 2])
def test_empty_support_stops_with_finite_action_values(horizon: int) -> None:
    problem = Problem(4)
    x = np.empty((0, 4), dtype=np.uint8)
    plans = fit_plans(
        problem, Dataset(x, np.empty(0, dtype=np.uint8), x.astype(bool))
    )
    for arm in ARMS:
        plan = plans[arm, horizon]
        assert plan.root_action == -1
        np.testing.assert_array_equal(plan.root_q, 0.5)
        assert problem.evaluate(plan) == 0.5


def test_myopic_policy_can_spend_budget_on_shortcut() -> None:
    problem = Problem(4)
    # Only the shortcut is observed, and it predicts every training label.
    x = problem.core.copy()
    available = np.zeros_like(x, dtype=bool)
    available[:, SHORTCUT] = True
    tables = build_count_tables(Dataset(x, x[:, SHORTCUT], available))
    plan = plan_model_based(
        tables, problem.costs, problem.fixed_predictor(), horizon=1
    )
    assert plan.root_action == SHORTCUT
    assert problem.evaluate(plan) == 0.75


def test_paired_results_are_reproducible_and_render(tmp_path: Path) -> None:
    results = run_study(reps=2, seed=0, jobs=1, smoke=True)
    assert results == run_study(reps=2, seed=0, jobs=2, smoke=True)
    assert len(results.regrets) == 3 * 3 * 2 * 4 * 2
    assert len(results.values) == 2 * len(results.regrets)
    assert {row.horizon for row in results.regrets} == {1, 2}
    assert {row.population_accuracy for row in results.regrets} == {0.75, 1}
    regrets, values = tmp_path / "regrets.csv", tmp_path / "values.csv"
    write_results(regrets, results.regrets)
    write_action_values(values, results.values)
    render_regrets(regrets, tmp_path / "regrets")
    for suffix in ("pdf", "png"):
        assert (tmp_path / f"regrets.{suffix}").stat().st_size > 0


def test_plot_rejects_legacy_or_unpaired_results(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text("d,p_miss,n,arm,rep,regret\n6,0.3,10,mask_local,0,0.5\n")
    with pytest.raises(ValueError, match="regenerate"):
        read_means(path)
    results = run_study(reps=1, seed=0, jobs=1, smoke=True)
    write_results(path, results.regrets[1:])
    with pytest.raises(ValueError, match="Unpaired"):
        read_means(path)


def test_action_value_confidence_interval_uses_replicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "values.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(exact_study.ActionValue._fields)
        for horizon in (1, 2):
            for rep, accuracy in enumerate((0.5, 0.75, 1.0)):
                writer.writerow(
                    (6, 0.3, 10, ARM_LOCAL, rep, horizon, CONTEXT, accuracy)
                )
    for curve in read_means(path, values=True).values():
        assert curve.mean[0] == 0.75
        assert curve.ci95[0] == pytest.approx(1.96 * 0.25 / np.sqrt(3))


def test_conceptual_constants_match_the_displayed_mask() -> None:
    constants = render()

    assert AVAILABILITY.all(axis=1).sum() == 1
    assert AVAILABILITY[:, :2].all(axis=1).sum() == 3
    assert r"\newcommand{\peQTrainContextTwo}{0.50}" in constants
    assert r"\newcommand{\planeFloor}{0.25}" in constants


def test_theoretical_arms_have_public_paper_labels() -> None:
    assert PANELS == (
        ("mask_agnostic", "(a) Aliasing"),
        ("mask_local", "(b) Filtering"),
        ("generative", "(c) Generative Restoration"),
    )


def test_collapse_plot_rescales_by_each_approach_effective_size() -> None:
    key = Curve("mask_local", 8, 0.5, 2)
    n = np.array([256, 1024])
    np.testing.assert_allclose(effective_size("mask_local", n, key), [1, 4])
    np.testing.assert_allclose(effective_size("generative", n, key), [64, 256])
