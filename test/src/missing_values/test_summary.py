from pathlib import Path

import pandas as pd

from scripts.analysis.summarize_missing_data import (
    _add_complete_data_gap,
    _aggregate,
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


def test_missing_data_summary_extracts_terminal_metrics_and_actions(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "method-aaco+mechanism-mcar+p-0.5+strategy-restricted+instance-0"
        / "eval_data.parquet"
    )
    _write_evaluation(path)

    metrics, actions = _instance_metrics(path)

    assert metrics["accuracy"] == 0.5
    assert metrics["mean_selections"] == 0.5
    assert metrics["mean_cost"] == 0.5
    assert actions[0]["selection"] == 0
    assert actions[0]["acquisitions_per_sample"] == 0.5


def test_missing_data_summary_uses_base_method_complete_reference() -> None:
    instances = pd.DataFrame.from_records(
        [
            {
                "method": "aaco",
                "mechanism": "none",
                "p": 0.0,
                "strategy": "complete",
                "strategy_display_name": "Complete data",
                "instance": 0,
                "accuracy": 0.9,
                "mean_selections": 14.0,
                "mean_cost": 14.0,
                "forced_stop_rate": 1.0,
            },
            {
                "method": "aaco_doubly_robust",
                "mechanism": "mcar",
                "p": 0.5,
                "strategy": "restricted",
                "strategy_display_name": "Restricted",
                "instance": 0,
                "accuracy": 0.8,
                "mean_selections": 10.0,
                "mean_cost": 10.0,
                "forced_stop_rate": 0.0,
            },
        ]
    )

    with_gap = _add_complete_data_gap(instances)
    summary = _aggregate(with_gap)

    restricted = with_gap.loc[with_gap["strategy"] == "restricted"].iloc[0]
    assert abs(restricted["accuracy_gap_to_complete"] + 0.1) < 1e-10
    assert len(summary) == 2
