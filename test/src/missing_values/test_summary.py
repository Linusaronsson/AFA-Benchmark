from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.summarize_missing_data import (
    _add_complete_data_gaps,
    _aggregate,
    _evaluation_paths,
    _instance_metrics,
)


def _write_evaluation(path: Path) -> None:
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "idx": 0,
                "prev_selections_performed": [],
                "action_performed": 1,
                "external_predicted_class": 0,
                "true_class": 0,
                "accumulated_cost": 1.0,
                "forced_stop": False,
            },
            {
                "idx": 0,
                "prev_selections_performed": [0],
                "action_performed": 0,
                "external_predicted_class": 0,
                "true_class": 0,
                "accumulated_cost": 1.0,
                "forced_stop": False,
            },
            {
                "idx": 1,
                "prev_selections_performed": [],
                "action_performed": 0,
                "external_predicted_class": 1,
                "true_class": 0,
                "accumulated_cost": 0.0,
                "forced_stop": False,
            },
        ]
    ).to_parquet(path, index=False)


def test_missing_data_summary_extracts_dataset_and_terminal_metrics(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "dataset-cube"
        / (
            "method-aaco+mechanism-mcar+p-0.5+strategy-restricted+"
            "instance-0+train_hard_budget-10+eval_hard_budget-10"
        )
        / "eval_data.parquet"
    )
    _write_evaluation(path)

    metrics, actions = _instance_metrics(path)

    assert metrics["dataset"] == "cube"
    assert metrics["accuracy"] == 0.5
    assert metrics["mean_selections"] == 0.5
    assert metrics["mean_cost"] == 0.5
    assert actions[0]["selection"] == 0
    assert actions[0]["acquisitions_per_sample"] == 0.5


def _metric_row(
    dataset: str,
    method: str,
    strategy: str,
    accuracy: float,
    f_score: float,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "method": method,
        "mechanism": "none" if strategy == "complete" else "mcar",
        "p": 0.0 if strategy == "complete" else 0.5,
        "strategy": strategy,
        "strategy_display_name": strategy,
        "instance": 0,
        "train_hard_budget": 10.0,
        "eval_hard_budget": 10.0,
        "accuracy": accuracy,
        "f_score": f_score,
        "mean_selections": 10.0,
        "mean_cost": 10.0,
        "forced_stop_rate": 0.0,
    }


def test_complete_references_are_scoped_by_dataset_and_base_method() -> None:
    instances = pd.DataFrame.from_records(
        [
            _metric_row("cube", "aaco", "complete", 0.9, 0.8),
            _metric_row("bank_marketing", "aaco", "complete", 0.6, 0.5),
            _metric_row("cube", "aaco_doubly_robust", "restricted", 0.8, 0.7),
            _metric_row(
                "bank_marketing",
                "aaco_doubly_robust",
                "restricted",
                0.55,
                0.45,
            ),
        ]
    )

    with_gaps = _add_complete_data_gaps(
        instances, {"aaco_doubly_robust": "aaco"}
    )
    summary = _aggregate(with_gaps)

    restricted = with_gaps.loc[with_gaps["strategy"] == "restricted"]
    cube = restricted.loc[restricted["dataset"] == "cube"].iloc[0]
    bank = restricted.loc[restricted["dataset"] == "bank_marketing"].iloc[0]
    assert cube["accuracy_gap_to_complete"] == pytest.approx(-0.1)
    assert bank["f_score_gap_to_complete"] == pytest.approx(-0.05)
    assert set(summary["dataset"]) == {"cube", "bank_marketing"}


def test_dataset_scoped_discovery_excludes_legacy_layout(
    tmp_path: Path,
) -> None:
    current = (
        tmp_path
        / "dataset-cube"
        / (
            "method-aaco+mechanism-mcar+p-0.5+strategy-restricted+"
            "instance-0+train_hard_budget-10+eval_hard_budget-10"
        )
        / "eval_data.parquet"
    )
    legacy = (
        tmp_path
        / "method-aaco+mechanism-mcar+p-0.5+strategy-restricted+instance-0"
        / "eval_data.parquet"
    )
    _write_evaluation(current)
    _write_evaluation(legacy)

    assert _evaluation_paths(tmp_path) == [current]
