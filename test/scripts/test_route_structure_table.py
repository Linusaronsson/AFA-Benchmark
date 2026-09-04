from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from scripts.analysis.route_structure_table import (
    PAPER_DATASETS,
    collect,
    render,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_collect_keeps_largest_budget_and_orders_by_overlap(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "dataset": dataset,
            "instance": instance,
            "budget": budget,
            "static_reference_score": 0.7 + 0.01 * instance,
            "route_sensitivity": 0.2 + 0.01 * instance,
            "weighted_route_overlap": overlap + 0.01 * instance,
        }
        for dataset, budget, overlap in (
            ("first", 1.0, 0.9),
            ("first", 2.0, 0.3),
            ("second", 1.0, 0.8),
        )
        for instance in range(2)
    ]
    path = tmp_path / "routes.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    frame = collect([path])

    assert frame["dataset"].tolist() == ["second", "first"]
    assert frame.set_index("dataset").loc["first", "budget"] == 2


def test_render_reports_se_and_bolds_only_clear_extremes() -> None:
    frame = pd.DataFrame(
        [
            {
                "dataset": "first",
                "budget": 3,
                "v_static": 0.2,
                "v_static_sem": 0.01,
                "route_sensitivity": 0.7,
                "route_sensitivity_sem": 0.01,
                "weighted_route_overlap": 0.9,
                "weighted_route_overlap_sem": 0.01,
            },
            {
                "dataset": "second",
                "budget": 3,
                "v_static": 0.5,
                "v_static_sem": 0.01,
                "route_sensitivity": 0.4,
                "route_sensitivity_sem": 0.01,
                "weighted_route_overlap": 0.6,
                "weighted_route_overlap_sem": 0.01,
            },
            {
                "dataset": "third",
                "budget": 3,
                "v_static": 0.6,
                "v_static_sem": 0.01,
                "route_sensitivity": 0.2,
                "route_sensitivity_sem": 0.01,
                "weighted_route_overlap": 0.5,
                "weighted_route_overlap_sem": 0.01,
            },
        ]
    )

    latex = render(frame)

    assert "$\\boldsymbol{0.200 \\pm 0.010}$" in latex
    assert "$\\boldsymbol{0.700 \\pm 0.010}$" in latex
    assert "$\\boldsymbol{0.900 \\pm 0.010}$" in latex
    assert "$0.500 \\pm 0.010$" in latex


def test_collect_requires_every_input_path(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="missing route CSVs"):
        collect([missing])


def test_collect_rejects_duplicate_route_cells(tmp_path: Path) -> None:
    row = {
        "dataset": "first",
        "instance": 0,
        "budget": 1.0,
        "static_reference_score": 0.7,
        "route_sensitivity": 0.2,
        "weighted_route_overlap": 0.5,
    }
    path = tmp_path / "routes.csv"
    pd.DataFrame([row, row]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate route cells"):
        collect([path])


def test_paper_protocol_requires_declared_k_splits_and_instances(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "dataset": dataset,
            "instance": instance,
            "budget": 10.0,
            "selection_split": "train",
            "eval_split": "val",
            "metric": "accuracy",
            "k_requested": 2000,
            "k_unique": 100,
            "static_reference_cost": 10.0,
            "static_reference_score": 0.7,
            "empty_route_selection_score": 0.5,
            "route_sensitivity": 0.2,
            "random_route_score_mean": 0.5,
            "weighted_route_overlap": 0.5,
        }
        for dataset in PAPER_DATASETS
        for instance in range(5)
    ]
    path = tmp_path / "routes.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    frame = collect(
        [path],
        expected_datasets=PAPER_DATASETS,
        expected_instances=5,
        expected_k=2000,
        selection_split="train",
        eval_split="val",
    )
    assert set(frame["dataset"]) == PAPER_DATASETS

    rows.pop()
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match="instances"):
        collect(
            [path],
            expected_datasets=PAPER_DATASETS,
            expected_instances=5,
            expected_k=2000,
            selection_split="train",
            eval_split="val",
        )
