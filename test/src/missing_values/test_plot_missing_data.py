from pathlib import Path

import pandas as pd

from afabench.plotting.config import PlottingDisplayConfig
from scripts.plotting.plot_missing_data import (
    _aggregate_action_rates,
    _prepare_performance_frame,
    generate_missing_data_plots,
)

DATASETS = ["cube", "bank_marketing"]


def _plotting_config() -> PlottingDisplayConfig:
    return PlottingDisplayConfig(
        plot_width=8.0,
        plot_height=4.0,
        plot_font_family="DejaVu Serif",
        method_name_mapping={
            "aaco": "AACO",
            "aaco_doubly_robust": "AACO (doubly robust)",
        },
        method_policy_family_mapping={
            "aaco": "aaco",
            "aaco_doubly_robust": "aaco",
        },
        method_family_color_schemes={"test": {"aaco": "#A6761D"}},
        active_method_color_scheme="test",
        # A method and its reweighting control share a family, so without an
        # override they would share a colour and draw as one line.
        method_color_overrides={"aaco_doubly_robust": "#AA3377"},
        dataset_name_mapping={
            "cube": "CUBE",
            "bank_marketing": "BankMarketing",
        },
        datasets_with_f_score=["bank_marketing"],
        dataset_sets={},
        color_palette_name="Dark2",
    )


def _instance_metrics() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for instance in [0, 1]:
            for method, strategy in [
                ("aaco", "restricted"),
                ("aaco", "mean_fill"),
                ("aaco_doubly_robust", "restricted"),
            ]:
                rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "mechanism": "mcar",
                        "p": 0.5,
                        "strategy": strategy,
                        "instance": instance,
                        "strategy_display_name": strategy,
                    }
                )
    return pd.DataFrame.from_records(rows)


def _summary() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        for method, strategy, value, sem in [
            ("aaco", "restricted", 0.7, 0.02),
            ("aaco", "mean_fill", 0.75, float("nan")),
            ("aaco_doubly_robust", "restricted", 0.72, 0.01),
        ]:
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "mechanism": "mcar",
                    "p": 0.5,
                    "strategy": strategy,
                    "strategy_display_name": strategy,
                    "accuracy_mean": value,
                    "accuracy_sem": sem,
                    "f_score_mean": value - 0.1,
                    "f_score_sem": sem,
                    "accuracy_gap_to_complete_mean": value - 0.8,
                    "accuracy_gap_to_complete_sem": sem,
                    "f_score_gap_to_complete_mean": value - 0.75,
                    "f_score_gap_to_complete_sem": sem,
                }
            )
    return pd.DataFrame.from_records(rows)


def _action_rates() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS:
        rows.extend(
            [
                {
                    "dataset": dataset,
                    "method": "aaco",
                    "mechanism": "mcar",
                    "p": 0.5,
                    "strategy": "restricted",
                    "instance": 0,
                    "strategy_display_name": "Restricted-action training",
                    "selection": 0,
                    "acquisitions_per_sample": 0.6,
                },
                {
                    "dataset": dataset,
                    "method": "aaco",
                    "mechanism": "mcar",
                    "p": 0.5,
                    "strategy": "mean_fill",
                    "instance": 0,
                    "strategy_display_name": "Mean completion",
                    "selection": 1,
                    "acquisitions_per_sample": 0.4,
                },
            ]
        )
    return pd.DataFrame.from_records(rows)


def _restoration_rmse() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "dataset": dataset,
                "mechanism": "mcar",
                "p": 0.5,
                "instance": instance,
                "strategy": "pvae_label_free",
                "strategy_display_name": "PVAE (label-free)",
                "split": split,
                "imputation_rmse": 0.2 + 0.01 * instance,
            }
            for dataset in DATASETS
            for instance in [0, 1]
            for split in ["train", "val"]
        ]
    )


def test_action_aggregation_is_dataset_scoped_and_fills_zeros() -> None:
    aggregated, _ = _aggregate_action_rates(
        _instance_metrics(),
        _action_rates(),
        "cube",
        "mcar",
        0.5,
        [0],
    )

    row = aggregated.loc[
        (aggregated["method"] == "aaco")
        & (aggregated["strategy"] == "restricted")
    ].iloc[0]
    assert row["acquisitions_per_sample"] == 0.3


def test_performance_frame_replaces_nan_sem() -> None:
    frame, _ = _prepare_performance_frame(
        _summary(), "cube", "mcar", "accuracy", _plotting_config()
    )
    mean_fill = frame.loc[frame["strategy"] == "mean_fill"].iloc[0]

    assert mean_fill["low_metric"] == mean_fill["mean_metric"]
    assert mean_fill["high_metric"] == mean_fill["mean_metric"]


def test_plots_are_written_per_dataset_without_pooled_output(
    tmp_path: Path,
) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    frames = {
        "instance_metrics": _instance_metrics(),
        "summary": _summary(),
        "action_rates": _action_rates(),
        "restoration_rmse": _restoration_rmse(),
    }
    paths = {}
    for name, frame in frames.items():
        paths[name] = inputs / f"{name}.csv"
        frame.to_csv(paths[name], index=False)

    output = tmp_path / "figures"
    generate_missing_data_plots(
        paths["instance_metrics"],
        paths["summary"],
        paths["action_rates"],
        paths["restoration_rmse"],
        output,
        _plotting_config(),
        ["svg"],
    )

    expected = {
        output / "dataset-cube" / "performance" / "accuracy_mcar.svg",
        output
        / "dataset-cube"
        / "performance"
        / "accuracy_gap_to_complete_mcar.svg",
        output
        / "dataset-cube"
        / "actions"
        / "acquisition_rate_mcar_p-0.5.svg",
        output / "dataset-cube" / "restoration" / "restoration_rmse.svg",
        output / "dataset-bank_marketing" / "performance" / "f_score_mcar.svg",
        output
        / "dataset-bank_marketing"
        / "performance"
        / "f_score_gap_to_complete_mcar.svg",
        output
        / "dataset-bank_marketing"
        / "actions"
        / "acquisition_rate_mcar_p-0.5.svg",
        output
        / "dataset-bank_marketing"
        / "restoration"
        / "restoration_rmse.svg",
    }
    assert set(output.glob("**/*.svg")) == expected
    assert all(path.stat().st_size > 0 for path in expected)
