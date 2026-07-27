from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from afabench.components.unmaskers.cube_nm_unmasker import CubeNMUnmasker
from scripts.analysis.route_redundancy import (
    compute_effects,
    gate_summary,
    planning_gate_summary,
    primary_metric,
    route_metrics,
    sample_feasible_routes,
    selection_feature_masks,
)


def test_cube_nm_routes_use_grouped_selection_space() -> None:
    unmasker = CubeNMUnmasker(n_contexts=5)
    masks = selection_feature_masks(
        unmasker, torch.Size([8]), device=torch.device("cpu")
    )
    costs = unmasker.get_selection_costs(torch.ones(8))

    assert masks.shape == (4, 8)
    assert masks[0].tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert masks[1].tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert costs.tolist() == [5.0, 1.0, 1.0, 1.0]


def test_random_routes_are_cost_feasible() -> None:
    costs = np.asarray([5.0, 1.0, 2.0, 4.0])
    routes = sample_feasible_routes(
        costs, budget=6.0, k=50, generator=np.random.default_rng(7)
    )

    assert routes
    assert all(costs[list(route)].sum() <= 6.0 for route in routes)
    assert all(tuple(sorted(route)) == route for route in routes)


def test_primary_metric_is_predeclared_by_dataset() -> None:
    assert primary_metric("actg") == "f_score"
    assert primary_metric("diabetes") == "f_score"
    assert primary_metric("physionet") == "f_score"
    assert primary_metric("cube_nm") == "accuracy"
    assert primary_metric("ckd") == "f_score"
    assert primary_metric("nhanes_mortality") == "f_score"


def test_top_routes_are_selected_without_evaluation_scores() -> None:
    correct = np.asarray(
        [[True, True, False], [True, False, False], [False, True, True]]
    )
    evaluation_scores = np.asarray([0.9, 0.2, 0.8])
    validation_scores = np.asarray([0.1, 0.9, 0.8])

    metrics = route_metrics(
        correct, evaluation_scores, validation_scores, top_frac=0.5
    )

    assert metrics["selected_sampled_route_score"] == 0.2


def _effect_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    route_rows = []
    complete_scores = {
        "aaco": 0.75,
        "ol_without_mask": 0.74,
        "dime": 0.72,
    }
    missing_scores = {
        0.3: {
            "aaco": {
                "restricted": 0.72,
                "pvae_label_conditioned": 0.74,
            },
            "ol_without_mask": {
                "restricted": 0.71,
                "pvae_label_conditioned": 0.73,
            },
        },
        0.7: {
            "aaco": {
                "restricted": 0.70,
                "pvae_label_conditioned": 0.73,
            },
            "ol_without_mask": {
                "restricted": 0.69,
                "pvae_label_conditioned": 0.71,
            },
        },
    }
    for instance in range(5):
        for budget in (3.0, 10.0):
            route_rows.append(
                {
                    "dataset": "actg",
                    "instance": instance,
                    "budget": budget,
                    "metric": "f_score",
                    "static_reference_score": 0.70,
                }
            )
            for method, accuracy in complete_scores.items():
                metric_rows.append(
                    {
                        "dataset": "actg",
                        "method": method,
                        "mechanism": "none",
                        "p": 0.0,
                        "strategy": "complete",
                        "instance": instance,
                        "eval_hard_budget": budget,
                        "accuracy": 0.1,
                        "f_score": accuracy,
                    }
                )
            for probability, method_scores in missing_scores.items():
                for method, strategies in method_scores.items():
                    for strategy, accuracy in strategies.items():
                        metric_rows.append(
                            {
                                "dataset": "actg",
                                "method": method,
                                "mechanism": "mcar",
                                "p": probability,
                                "strategy": strategy,
                                "instance": instance,
                                "eval_hard_budget": budget,
                                "accuracy": 0.1,
                                "f_score": accuracy,
                            }
                        )
    return pd.DataFrame(metric_rows), pd.DataFrame(route_rows)


def test_effects_keep_aaco_and_ol_separate() -> None:
    metrics, routes = _effect_inputs()

    planning, missingness = compute_effects(metrics, routes)

    assert set(planning["method"]) == {"aaco", "ol_without_mask"}
    assert planning["metric"].eq("f_score").all()
    assert missingness["metric"].eq("f_score").all()
    assert planning["eval_hard_budget"].eq(10.0).all()
    assert missingness["eval_hard_budget"].eq(10.0).all()
    assert set(missingness["p"]) == {0.3, 0.7}
    aaco = planning.loc[planning["method"] == "aaco"]
    ol = planning.loc[planning["method"] == "ol_without_mask"]
    assert np.allclose(aaco["adaptive_gain"], 0.05)
    assert np.allclose(aaco["nongreedy_gain"], 0.03)
    assert np.allclose(ol["adaptive_gain"], 0.04)
    assert np.allclose(ol["nongreedy_gain"], 0.02)

    aaco_missing = missingness.loc[
        (missingness["method"] == "aaco") & (missingness["p"] == 0.7)
    ]
    ol_missing = missingness.loc[
        (missingness["method"] == "ol_without_mask")
        & (missingness["p"] == 0.7)
    ]
    assert np.allclose(aaco_missing["missingness_damage"], 0.05)
    assert np.allclose(aaco_missing["restoration_gain"], 0.03)
    assert np.allclose(ol_missing["missingness_damage"], 0.05)
    assert np.allclose(ol_missing["restoration_gain"], 0.02)


def test_gate_requires_both_non_greedy_methods() -> None:
    metrics, routes = _effect_inputs()
    planning, missingness = compute_effects(metrics, routes)

    gate = gate_summary(planning, missingness)

    assert gate["method_pass"].all()
    assert gate["dataset_concordant"].all()

    missingness = missingness.loc[missingness["method"] == "aaco"]
    gate = gate_summary(planning, missingness)
    assert gate.loc[gate["method"] == "aaco", "method_pass"].all()
    assert not gate["dataset_concordant"].any()


def test_planning_gate_does_not_require_restoration_rows() -> None:
    metrics, routes = _effect_inputs()
    planning, _ = compute_effects(metrics, routes)

    gate = planning_gate_summary(planning)

    assert gate["planning_pass"].all()
    assert gate["dataset_concordant"].all()


def test_gates_require_all_five_instances() -> None:
    metrics, routes = _effect_inputs()
    planning, missingness = compute_effects(metrics, routes)
    planning = planning.loc[planning["instance"] != 4]
    missingness = missingness.loc[missingness["instance"] != 4]

    assert not planning_gate_summary(planning)["planning_pass"].any()
    assert not gate_summary(planning, missingness)["method_pass"].any()


def test_duplicate_cells_are_rejected() -> None:
    metrics, routes = _effect_inputs()
    metrics = pd.concat([metrics, metrics.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate exact cells"):
        compute_effects(metrics, routes)
