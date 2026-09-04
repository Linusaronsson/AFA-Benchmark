from __future__ import annotations

import numpy as np

from scripts.paper.conceptual_constants import AVAILABILITY, render
from scripts.paper.exact_study import (
    ARM_AGNOSTIC,
    ARM_GENERATIVE,
    ARM_LOCAL,
    BUDGET,
    SHORTCUT,
    Dataset,
    Problem,
    build_count_tables,
    plan_mask_agnostic,
    plan_model_based,
    run_study,
)
from scripts.paper.plot_exact_study import PANELS


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
    results = run_study(reps=1, seed=0, jobs=1, smoke=True)

    assert {result.arm for result in results} >= {
        ARM_LOCAL,
        ARM_AGNOSTIC,
        ARM_GENERATIVE,
    }
    assert all(0 <= result.regret <= 0.5 for result in results)


def test_conceptual_constants_match_the_displayed_mask() -> None:
    constants = render()

    assert AVAILABILITY.all(axis=1).sum() == 1
    assert AVAILABILITY[:, :2].all(axis=1).sum() == 3
    assert r"\newcommand{\peQTrainContextTwo}{0.50}" in constants
    assert r"\newcommand{\planeFloor}{0.25}" in constants


def test_theoretical_arms_have_public_paper_labels() -> None:
    assert dict(PANELS) == {
        "mask_local": "(a) Filtering",
        "mask_agnostic": "(b) Aliasing",
        "generative": "(c) Generative restoration",
    }
