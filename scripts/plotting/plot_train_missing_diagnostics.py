from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as p9
import polars as pl

from afabench.common.naming import LEGACY_DATASET_KEY_ALIASES
from afabench.common.parquet import (
    collect_streaming,
    require_columns,
    scan_parquet,
)
from afabench.eval.plotting_config import (
    DATASET_NAME_MAPPING,
    DATASETS_WITH_F_SCORE,
    METHOD_COLOR_MAPPING,
    METHOD_NAME_MAPPING,
    PLOT_WIDTH,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray


MECHANISM_LABELS = {
    "mcar": "MCAR",
    "mar": "MAR",
    "mnar_logistic": "MNAR (logistic)",
}

MECHANISM_ORDER = ["MCAR", "MAR", "MNAR (logistic)"]

CURVE_SPECS: dict[str, dict[str, str]] = {
    "gadgil2023": {
        "curve_label": "DIME + block-only",
        "reference_method": "gadgil2023",
        "policy_type": "Myopic",
    },
    "gadgil2023_ipw_feature_marginal": {
        "curve_label": "DIME + IPW",
        "reference_method": "gadgil2023",
        "policy_type": "Myopic",
    },
    "aaco_zero_fill": {
        "curve_label": "AACO + zero-fill",
        "reference_method": "aaco_full",
        "policy_type": "Non-myopic",
    },
    "aaco_mask_aware": {
        "curve_label": "AACO + mask-aware",
        "reference_method": "aaco_full",
        "policy_type": "Non-myopic",
    },
    "aaco_dr": {
        "curve_label": "AACO + DR",
        "reference_method": "aaco_full",
        "policy_type": "Non-myopic",
    },
    "ol_with_mask": {
        "curve_label": "OL-MFRL + block-only",
        "reference_method": "ol_with_mask",
        "policy_type": "Non-myopic",
    },
}

CURVE_LABEL_ORDER = [
    "DIME + block-only",
    "DIME + IPW",
    "AACO + zero-fill",
    "AACO + mask-aware",
    "AACO + DR",
    "OL-MFRL + block-only",
]

DATASET_GROUPS = {
    "cube_nm": "CUBE-NM family",
    "cube_nm_without_noise": "CUBE-NM family",
    # "cube": "CUBE family",
    # "cube_without_noise": "CUBE family",
    "actg": "Real-world",
    "ckd": "Real-world",
    "diabetes": "Real-world",
    "physionet": "Real-world",
}

# DATASET_GROUP_ORDER = ["CUBE-NM family", "CUBE family", "Real-world"]
DATASET_GROUP_ORDER = ["CUBE-NM family", "Real-world"]
DATASET_ORDER = [
    "cube_nm",
    "cube_nm_without_noise",
    "cube",
    "cube_without_noise",
    "actg",
    "ckd",
    "diabetes",
    "physionet",
]

BASELINE_INITIALIZER = "train_initializer-cold+eval_initializer-cold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate thesis diagnostics for training under missingness.",
    )
    parser.add_argument("input_path", type=Path, help="Merged eval parquet.")
    parser.add_argument("output_dir", type=Path, help="Output directory.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "svg", "png"],
        help="Output formats for figures.",
    )
    parser.add_argument(
        "--heatmap-dataset",
        default="cube_nm_without_noise",
        help="Dataset used for the action heatmap diagnostic.",
    )
    parser.add_argument(
        "--heatmap-method",
        default="ol_with_mask",
        help="Method used for the action heatmap diagnostic.",
    )
    parser.add_argument(
        "--heatmap-rate",
        type=float,
        default=0.7,
        help="Missingness rate used in the action heatmap diagnostic.",
    )
    return parser.parse_args()


def _parse_mechanism_and_rate(initializer: str) -> tuple[str, float] | None:
    match = re.fullmatch(
        r"train_initializer-(.+)\+eval_initializer-(.+)",
        initializer,
    )
    train_initializer = match.group(1) if match else initializer
    if train_initializer == "cold":
        return ("cold", 0.0)

    rate_match = re.fullmatch(
        r"(mcar|mar|mnar_logistic)_p(\d+)",
        train_initializer,
    )
    if rate_match is None:
        return None
    mechanism, rate_code = rate_match.groups()
    rate = float(rate_code) / (10 ** max(len(rate_code) - 1, 0))
    return mechanism, rate


def _curve_label(afa_method: str) -> str | None:
    spec = CURVE_SPECS.get(afa_method)
    return None if spec is None else spec["curve_label"]


def _reference_method(afa_method: str) -> str | None:
    spec = CURVE_SPECS.get(afa_method)
    return None if spec is None else spec["reference_method"]


def _policy_type(afa_method: str) -> str | None:
    spec = CURVE_SPECS.get(afa_method)
    return None if spec is None else spec["policy_type"]


def _display_dataset(dataset: str) -> str:
    metric = "F1" if dataset in DATASETS_WITH_F_SCORE else "Acc."
    return f"{DATASET_NAME_MAPPING.get(dataset, dataset)} ({metric})"


def _display_method_color(afa_method: str) -> str:
    method_name = METHOD_NAME_MAPPING.get(afa_method, afa_method)
    return METHOD_COLOR_MAPPING.get(method_name, "#333333")


LABEL_COLORS = {
    spec["curve_label"]: _display_method_color(method)
    for method, spec in CURVE_SPECS.items()
}


def _load_filtered(input_path: Path) -> pl.LazyFrame:
    data = scan_parquet(input_path)
    required_cols = {
        "action_performed",
        "predicted_class",
        "true_class",
        "dataset",
        "afa_method",
        "initializer",
        "train_seed",
        "eval_seed",
        "eval_hard_budget",
        "train_hard_budget",
    }
    require_columns(data, required_cols)

    return (
        data.select(sorted(required_cols))
        .with_columns(
            dataset=pl.col("dataset").replace(LEGACY_DATASET_KEY_ALIASES)
        )
        .filter(pl.col("eval_hard_budget").is_not_null())
        .filter(
            pl.col("eval_hard_budget")
            == pl.col("eval_hard_budget").max().over("dataset")
        )
    )


def _compute_metrics_per_run(data: pl.LazyFrame) -> pl.LazyFrame:
    group_cols = [
        "dataset",
        "afa_method",
        "initializer",
        "train_seed",
        "eval_seed",
        "eval_hard_budget",
        "train_hard_budget",
    ]
    final_predictions = data.filter(
        (pl.col("action_performed") == 0)
        & pl.col("predicted_class").is_not_null()
    )

    confusion = final_predictions.group_by(
        [*group_cols, "true_class", "predicted_class"]
    ).agg(n=pl.len())

    accuracy = (
        confusion.group_by(group_cols)
        .agg(
            total=pl.col("n").sum(),
            correct=pl.when(pl.col("true_class") == pl.col("predicted_class"))
            .then(pl.col("n"))
            .otherwise(0)
            .sum(),
        )
        .with_columns(
            accuracy=pl.col("correct").cast(pl.Float64)
            / pl.col("total").clip(lower_bound=1).cast(pl.Float64)
        )
        .select([*group_cols, "accuracy"])
    )

    true_totals = (
        confusion.group_by([*group_cols, "true_class"])
        .agg(true_total=pl.col("n").sum().cast(pl.UInt64))
        .rename({"true_class": "label"})
        .with_columns(pred_total=pl.lit(0, dtype=pl.UInt64))
        .select([*group_cols, "label", "true_total", "pred_total"])
    )
    pred_totals = (
        confusion.group_by([*group_cols, "predicted_class"])
        .agg(pred_total=pl.col("n").sum().cast(pl.UInt64))
        .rename({"predicted_class": "label"})
        .with_columns(true_total=pl.lit(0, dtype=pl.UInt64))
        .select([*group_cols, "label", "true_total", "pred_total"])
    )
    label_totals = (
        pl.concat([true_totals, pred_totals])
        .group_by([*group_cols, "label"])
        .agg(
            true_total=pl.col("true_total").sum(),
            pred_total=pl.col("pred_total").sum(),
        )
    )
    tp_by_label = (
        confusion.filter(pl.col("true_class") == pl.col("predicted_class"))
        .select([*group_cols, pl.col("true_class").alias("label"), "n"])
        .rename({"n": "tp"})
        .with_columns(tp=pl.col("tp").cast(pl.UInt64))
    )
    f_score = (
        label_totals.join(
            tp_by_label,
            on=[*group_cols, "label"],
            how="left",
            nulls_equal=True,
        )
        .with_columns(
            tp=pl.col("tp").fill_null(0),
            denom=(pl.col("true_total") + pl.col("pred_total")).cast(
                pl.Float64
            ),
        )
        .with_columns(
            f_score=pl.when(pl.col("denom") > 0)
            .then(2.0 * pl.col("tp").cast(pl.Float64) / pl.col("denom"))
            .otherwise(0.0)
        )
        .group_by(group_cols)
        .agg(f_score=pl.col("f_score").mean())
    )

    metrics = accuracy.join(
        f_score,
        on=group_cols,
        how="left",
        nulls_equal=True,
    )
    return metrics.with_columns(
        metric=pl.when(pl.col("dataset").is_in(DATASETS_WITH_F_SCORE))
        .then(pl.col("f_score"))
        .otherwise(pl.col("accuracy"))
    )


def _add_metadata(metrics: pl.DataFrame) -> pl.DataFrame:
    mechanisms: list[str | None] = []
    mechanism_labels: list[str | None] = []
    rates: list[float | None] = []
    curve_labels: list[str | None] = []
    reference_methods: list[str | None] = []
    policy_types: list[str | None] = []
    dataset_groups: list[str] = []
    dataset_labels: list[str] = []

    for row in metrics.iter_rows(named=True):
        parsed = _parse_mechanism_and_rate(row["initializer"])
        mechanism = parsed[0] if parsed is not None else None
        rate = parsed[1] if parsed is not None else None
        mechanisms.append(mechanism)
        mechanism_labels.append(
            "Cold"
            if mechanism == "cold"
            else MECHANISM_LABELS.get(mechanism or "", mechanism)
        )
        rates.append(rate)
        curve_labels.append(_curve_label(row["afa_method"]))
        reference_methods.append(_reference_method(row["afa_method"]))
        policy_types.append(_policy_type(row["afa_method"]))
        dataset_groups.append(DATASET_GROUPS.get(row["dataset"], "Other"))
        dataset_labels.append(_display_dataset(row["dataset"]))

    return metrics.with_columns(
        mechanism=pl.Series(mechanisms, dtype=pl.String),
        mechanism_label=pl.Series(mechanism_labels, dtype=pl.String),
        rate=pl.Series(rates, dtype=pl.Float64),
        curve_label=pl.Series(curve_labels, dtype=pl.String),
        reference_method=pl.Series(reference_methods, dtype=pl.String),
        policy_type=pl.Series(policy_types, dtype=pl.String),
        dataset_group=pl.Series(dataset_groups, dtype=pl.String),
        dataset_label=pl.Series(dataset_labels, dtype=pl.String),
    )


def _compute_delta_rows(metrics: pl.DataFrame) -> pl.DataFrame:
    enriched = _add_metadata(metrics)
    curve_rows = enriched.filter(
        (pl.col("mechanism") != "cold")
        & pl.col("curve_label").is_not_null()
        & pl.col("reference_method").is_not_null()
    )
    baseline_rows = (
        enriched.filter(pl.col("initializer") == BASELINE_INITIALIZER)
        .select(
            "dataset",
            "afa_method",
            "train_seed",
            "eval_seed",
            "eval_hard_budget",
            pl.col("metric").alias("reference_metric"),
        )
        .rename({"afa_method": "reference_method"})
    )
    return curve_rows.join(
        baseline_rows,
        on=[
            "dataset",
            "reference_method",
            "train_seed",
            "eval_seed",
            "eval_hard_budget",
        ],
        how="inner",
        nulls_equal=True,
    ).with_columns(
        delta=pl.col("metric") - pl.col("reference_metric"),
        baseline_percent=pl.when(pl.col("reference_metric") > 0)
        .then(100.0 * pl.col("metric") / pl.col("reference_metric"))
        .otherwise(None),
    )


def _summarize_delta(delta_rows: pl.DataFrame) -> pl.DataFrame:
    return (
        delta_rows.group_by(
            [
                "dataset",
                "dataset_label",
                "dataset_group",
                "afa_method",
                "curve_label",
                "policy_type",
                "mechanism",
                "mechanism_label",
                "rate",
            ]
        )
        .agg(
            mean_metric=pl.col("metric").mean(),
            mean_reference=pl.col("reference_metric").mean(),
            mean_delta=pl.col("delta").mean(),
            std_delta=pl.col("delta").std(),
            mean_baseline_percent=pl.col("baseline_percent").mean(),
            std_baseline_percent=pl.col("baseline_percent").std(),
            n_runs=pl.len(),
        )
        .with_columns(
            std_delta=pl.col("std_delta").fill_null(0.0),
            std_baseline_percent=pl.col("std_baseline_percent").fill_null(0.0),
        )
        .with_columns(
            low_delta=pl.col("mean_delta") - pl.col("std_delta"),
            high_delta=pl.col("mean_delta") + pl.col("std_delta"),
            low_baseline_percent=pl.col("mean_baseline_percent")
            - pl.col("std_baseline_percent"),
            high_baseline_percent=pl.col("mean_baseline_percent")
            + pl.col("std_baseline_percent"),
        )
    )


def _prepare_categories(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["curve_label"] = pd.Categorical(
        frame["curve_label"],
        categories=[
            label
            for label in CURVE_LABEL_ORDER
            if label in set(frame["curve_label"])
        ],
        ordered=True,
    )
    frame["mechanism_label"] = pd.Categorical(
        frame["mechanism_label"],
        categories=[
            label
            for label in MECHANISM_ORDER
            if label in set(frame["mechanism_label"])
        ],
        ordered=True,
    )
    frame["dataset_group"] = pd.Categorical(
        frame["dataset_group"],
        categories=[
            group
            for group in DATASET_GROUP_ORDER
            if group in set(frame["dataset_group"])
        ],
        ordered=True,
    )
    dataset_label_order = [
        _display_dataset(dataset)
        for dataset in DATASET_ORDER
        if _display_dataset(dataset) in set(frame["dataset_label"])
    ]
    frame["dataset_label"] = pd.Categorical(
        frame["dataset_label"],
        categories=dataset_label_order,
        ordered=True,
    )
    return frame


def _save_plot(
    plot: p9.ggplot,
    output_path: Path,
    *,
    width: float,
    height: float,
    formats: Sequence[str],
) -> None:
    for fmt in formats:
        plot.save(
            output_path.with_suffix(f".{fmt}"),
            width=width,
            height=height,
        )


def _aggregate_dataset_means(
    plot_df: pd.DataFrame,
    value_column: str,
    mean_column: str,
    low_column: str,
    high_column: str,
) -> pd.DataFrame:
    aggregate = cast(
        "pd.DataFrame",
        plot_df.groupby(
            [
                "dataset_group",
                "mechanism_label",
                "rate",
                "curve_label",
                "policy_type",
            ],
            observed=True,
            as_index=False,
        )[value_column].agg(["mean", "std", "count"]),
    )
    aggregate = aggregate.rename(
        columns={
            "mean": mean_column,
            "std": "std_dataset",
            "count": "n_datasets",
        },
    )
    aggregate["std_dataset"] = aggregate["std_dataset"].fillna(0.0)
    aggregate["ci95"] = (
        1.96
        * aggregate["std_dataset"]
        / np.sqrt(aggregate["n_datasets"].clip(lower=1))
    )
    aggregate[low_column] = aggregate[mean_column] - aggregate["ci95"]
    aggregate[high_column] = aggregate[mean_column] + aggregate["ci95"]
    return aggregate


def _bootstrap_mean_ci(
    values: pd.Series,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20260508,
) -> tuple[float, float, float]:
    array = values.dropna().to_numpy(dtype=float)
    if len(array) == 0:
        return (float("nan"), float("nan"), float("nan"))
    if len(array) == 1:
        value = float(array[0])
        return (value, value, value)

    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_bootstrap, len(array)), replace=True)
    means = samples.mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _aggregate_family_bootstrap(
    plot_df: pd.DataFrame,
    *,
    rate: float = 0.7,
) -> pd.DataFrame:
    rate_df = plot_df[np.isclose(plot_df["rate"].astype(float), rate)].copy()
    rate_df["relative_change_percent"] = (
        rate_df["mean_baseline_percent"] - 100.0
    )

    rows: list[dict[str, object]] = []
    group_cols = ["dataset_group", "mechanism_label", "policy_type"]
    for key, group in rate_df.groupby(group_cols, observed=True):
        mean, low, high = _bootstrap_mean_ci(
            cast("pd.Series", group["relative_change_percent"]),
            seed=20260508 + len(rows),
        )
        dataset_group, mechanism_label, policy_type = cast(
            "tuple[object, object, object]", key
        )
        rows.append(
            {
                "dataset_group": dataset_group,
                "mechanism_label": mechanism_label,
                "policy_type": policy_type,
                "mean_relative_change_percent": mean,
                "low_relative_change_percent": low,
                "high_relative_change_percent": high,
                "n_rows": len(group),
            }
        )
    return pd.DataFrame(rows)


def plot_delta_by_rate(
    summary: pl.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    plot_df = _prepare_categories(
        summary.filter(pl.col("dataset_group") != "Other").to_pandas()
    )
    aggregate = _aggregate_dataset_means(
        plot_df,
        "mean_delta",
        "mean_delta",
        "low_delta",
        "high_delta",
    )

    labels = [
        label
        for label in CURVE_LABEL_ORDER
        if label in set(plot_df["curve_label"])
    ]
    plot = (
        p9.ggplot(aggregate, p9.aes(x="rate", y="mean_delta"))
        + p9.geom_hline(
            yintercept=0.0, color="#3a3a3a", linetype="dashed", size=0.7
        )
        + p9.geom_errorbar(
            p9.aes(ymin="low_delta", ymax="high_delta", color="curve_label"),
            width=0.025,
            alpha=0.65,
            show_legend=False,
        )
        + p9.geom_line(
            p9.aes(
                color="curve_label",
                linetype="policy_type",
                group="curve_label",
            ),
            size=1.25,
        )
        + p9.geom_point(
            p9.aes(color="curve_label", shape="policy_type"),
            size=2.4,
        )
        + p9.facet_grid("dataset_group ~ mechanism_label", scales="free_y")
        + p9.scale_color_manual(
            name="Policy",
            values={label: LABEL_COLORS[label] for label in labels},
            breaks=labels,
            limits=labels,
        )
        + p9.scale_shape_manual(
            name="Policy type",
            values={"Myopic": "o", "Non-myopic": "^"},
            breaks=["Myopic", "Non-myopic"],
            limits=["Myopic", "Non-myopic"],
        )
        + p9.scale_linetype_manual(
            name="Policy type",
            values={"Myopic": "solid", "Non-myopic": "dotted"},
            breaks=["Myopic", "Non-myopic"],
            limits=["Myopic", "Non-myopic"],
        )
        + p9.scale_x_continuous(
            breaks=[0.3, 0.5, 0.7],
            labels=["30%", "50%", "70%"],
        )
        + p9.labs(
            x="Training missingness rate",
            y="Metric difference from full-data baseline (95% CI)",
        )
        + p9.guides(
            color=p9.guide_legend(order=1),
            shape=p9.guide_legend(order=2),
            linetype=p9.guide_legend(order=2),
        )
        + p9.theme_bw()
        + p9.theme(figure_size=(PLOT_WIDTH, 7.5))
    )
    _save_plot(
        plot,
        output_dir / "train_missing_delta_by_rate",
        width=PLOT_WIDTH,
        height=7.5,
        formats=formats,
    )


def plot_baseline_percent_by_rate(
    summary: pl.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    plot_df = _prepare_categories(
        summary.filter(pl.col("dataset_group") != "Other").to_pandas()
    )
    aggregate = _aggregate_dataset_means(
        plot_df,
        "mean_baseline_percent",
        "mean_baseline_percent",
        "low_baseline_percent",
        "high_baseline_percent",
    )

    labels = [
        label
        for label in CURVE_LABEL_ORDER
        if label in set(plot_df["curve_label"])
    ]
    plot = (
        p9.ggplot(aggregate, p9.aes(x="rate", y="mean_baseline_percent"))
        + p9.geom_hline(
            yintercept=100.0, color="#3a3a3a", linetype="dashed", size=0.7
        )
        + p9.geom_errorbar(
            p9.aes(
                ymin="low_baseline_percent",
                ymax="high_baseline_percent",
                color="curve_label",
            ),
            width=0.025,
            alpha=0.65,
            show_legend=False,
        )
        + p9.geom_line(
            p9.aes(
                color="curve_label",
                linetype="policy_type",
                group="curve_label",
            ),
            size=1.25,
        )
        + p9.geom_point(
            p9.aes(color="curve_label", shape="policy_type"),
            size=2.4,
        )
        + p9.facet_grid("dataset_group ~ mechanism_label", scales="free_y")
        + p9.scale_color_manual(
            name="Policy",
            values={label: LABEL_COLORS[label] for label in labels},
            breaks=labels,
            limits=labels,
        )
        + p9.scale_shape_manual(
            name="Policy type",
            values={"Myopic": "o", "Non-myopic": "^"},
            breaks=["Myopic", "Non-myopic"],
            limits=["Myopic", "Non-myopic"],
        )
        + p9.scale_linetype_manual(
            name="Policy type",
            values={"Myopic": "solid", "Non-myopic": "dotted"},
            breaks=["Myopic", "Non-myopic"],
            limits=["Myopic", "Non-myopic"],
        )
        + p9.scale_x_continuous(
            breaks=[0.3, 0.5, 0.7],
            labels=["30%", "50%", "70%"],
        )
        + p9.scale_y_continuous(labels=lambda ys: [f"{y:.0f}%" for y in ys])
        + p9.labs(
            x="Training missingness rate",
            y="% of full-data baseline (95% CI)",
        )
        + p9.guides(
            color=p9.guide_legend(order=1),
            shape=p9.guide_legend(order=2),
            linetype=p9.guide_legend(order=2),
        )
        + p9.theme_bw()
        + p9.theme(figure_size=(PLOT_WIDTH, 7.5))
    )
    _save_plot(
        plot,
        output_dir / "train_missing_baseline_percent_by_rate_ci",
        width=PLOT_WIDTH,
        height=7.5,
        formats=formats,
    )


def plot_family_relative_change_at_70(
    delta_rows: pl.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    plot_df = _prepare_categories(
        delta_rows.filter(pl.col("dataset_group") != "Other").to_pandas()
    )
    plot_df["mean_baseline_percent"] = plot_df["baseline_percent"]
    aggregate = _aggregate_family_bootstrap(plot_df, rate=0.7)
    if aggregate.empty:
        return

    aggregate = aggregate.copy()
    aggregate["mechanism_label"] = pd.Categorical(
        aggregate["mechanism_label"],
        categories=[
            label
            for label in MECHANISM_ORDER
            if label in set(aggregate["mechanism_label"])
        ],
        ordered=True,
    )
    aggregate["dataset_group"] = pd.Categorical(
        aggregate["dataset_group"],
        categories=[
            group
            for group in DATASET_GROUP_ORDER
            if group in set(aggregate["dataset_group"])
        ],
        ordered=True,
    )

    # Numeric y positions so both tick labels are guaranteed to render
    # (a categorical y-axis was silently dropping the "Non-myopic" label).
    y_positions = {"Non-myopic": 0.0, "Myopic": 1.0}
    aggregate["y_pos"] = cast("pd.Series", aggregate["policy_type"]).map(
        y_positions
    )

    rate_points = cast(
        "pd.DataFrame",
        plot_df[np.isclose(plot_df["rate"].astype(float), 0.7)].copy(),
    )
    rate_points["relative_change_percent"] = (
        rate_points["mean_baseline_percent"] - 100.0
    )
    rate_points["y_pos"] = cast("pd.Series", rate_points["policy_type"]).map(
        y_positions
    )

    fig_height = 3.2
    plot = (
        p9.ggplot(
            aggregate,
            p9.aes(
                y="y_pos",
                x="mean_relative_change_percent",
                color="policy_type",
            ),
        )
        + p9.geom_vline(
            xintercept=0.0, color="#3a3a3a", linetype="dashed", size=0.6
        )
        + p9.geom_jitter(
            data=rate_points,
            mapping=p9.aes(
                y="y_pos",
                x="relative_change_percent",
                color="policy_type",
            ),
            height=0.10,
            width=0.0,
            alpha=0.18,
            size=1.2,
            show_legend=False,
        )
        + p9.geom_errorbarh(
            p9.aes(
                xmin="low_relative_change_percent",
                xmax="high_relative_change_percent",
            ),
            height=0.0,
            size=1.4,
            show_legend=False,
        )
        + p9.geom_point(size=3.2, show_legend=False)
        + p9.facet_grid("dataset_group ~ mechanism_label", scales="free_x")
        + p9.scale_color_manual(
            values={"Myopic": "#3366AA", "Non-myopic": "#CC6677"},
        )
        + p9.scale_y_continuous(
            breaks=[0.0, 1.0],
            labels=["Non-myopic", "Myopic"],
            limits=(-0.55, 1.55),
        )
        + p9.scale_x_continuous(labels=lambda xs: [f"{x:.0f}%" for x in xs])
        + p9.labs(
            y="",
            x="Relative change from full-data baseline at 70% missingness",
        )
        + p9.theme_bw()
        + p9.theme(
            figure_size=(PLOT_WIDTH, fig_height),
            axis_text_y=p9.element_text(rotation=0, ha="right"),
            legend_position="none",
        )
    )
    _save_plot(
        plot,
        output_dir / "train_missing_family_relative_change_p07_ci",
        width=PLOT_WIDTH,
        height=fig_height,
        formats=formats,
    )
    aggregate.to_csv(
        output_dir / "train_missing_family_relative_change_p07_ci.csv",
        index=False,
    )


def plot_delta_heatmap(
    summary: pl.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    heatmap_df = summary.filter(
        (pl.col("rate") == 0.7) & (pl.col("dataset_group") != "Other")
    ).to_pandas()
    if heatmap_df.empty:
        return
    heatmap_df = _prepare_categories(heatmap_df)
    heatmap_df["delta_label"] = heatmap_df["mean_delta"].map(
        lambda x: f"{x:+.2f}"
    )

    labels = [
        label
        for label in CURVE_LABEL_ORDER
        if label in set(heatmap_df["curve_label"])
    ]
    plot = (
        p9.ggplot(
            heatmap_df,
            p9.aes(x="curve_label", y="dataset_label", fill="mean_delta"),
        )
        + p9.geom_tile(color="white", size=0.55)
        + p9.geom_text(p9.aes(label="delta_label"), size=7)
        + p9.facet_wrap("mechanism_label", nrow=1)
        + p9.scale_x_discrete(limits=labels)
        + p9.scale_fill_gradient2(
            name="Metric difference",
            low="#b2182b",
            mid="#f7f7f7",
            high="#2166ac",
            midpoint=0.0,
        )
        + p9.labs(x="", y="")
        + p9.theme(
            figure_size=(PLOT_WIDTH, 5.2),
            axis_text_x=p9.element_text(rotation=35, ha="right"),
        )
    )
    _save_plot(
        plot,
        output_dir / "train_missing_delta_heatmap_p07",
        width=PLOT_WIDTH,
        height=5.2,
        formats=formats,
    )


def _initializer_for(mechanism: str, rate: float) -> str:
    rate_code = f"p{round(rate * 10):02d}"
    return f"train_initializer-{mechanism}_{rate_code}+eval_initializer-cold"


def _load_action_rows(
    input_path: Path,
    *,
    dataset: str,
    method: str,
    rate: float,
) -> pl.DataFrame:
    initializers = [
        BASELINE_INITIALIZER,
        _initializer_for("mcar", rate),
        _initializer_for("mar", rate),
        _initializer_for("mnar_logistic", rate),
    ]
    data = scan_parquet(input_path)
    required_cols = {
        "action_performed",
        "dataset",
        "afa_method",
        "initializer",
        "n_selections_performed",
        "eval_hard_budget",
    }
    require_columns(data, required_cols)
    return collect_streaming(
        data.select(sorted(required_cols))
        .with_columns(
            dataset=pl.col("dataset").replace(LEGACY_DATASET_KEY_ALIASES)
        )
        .filter(pl.col("dataset") == dataset)
        .filter(pl.col("afa_method") == method)
        .filter(pl.col("initializer").is_in(initializers))
        .filter(pl.col("eval_hard_budget").is_not_null())
        .filter(
            pl.col("eval_hard_budget")
            == pl.col("eval_hard_budget").max().over("dataset")
        )
        .filter(pl.col("action_performed") != 0)
    )


def _normalize_heatmap(
    df: pl.DataFrame,
    max_action: int,
    max_time: int,
) -> NDArray[np.float64]:
    heatmap = np.zeros((max_action, max_time + 1), dtype=float)
    for row in df.iter_rows(named=True):
        action = int(row["action_performed"])
        time_step = int(row["n_selections_performed"])
        if action > 0:
            heatmap[action - 1, time_step] += 1
    time_counts = np.bincount(
        df["n_selections_performed"].cast(pl.Int64).to_numpy(),
        minlength=max_time + 1,
    )
    time_counts = np.maximum(time_counts, 1)
    return heatmap / time_counts


def _format_action_axis(
    ax: Axes,
    *,
    title: str,
    max_action: int,
    max_time: int,
) -> None:
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)
    ax.set_xlabel("Time step", fontsize=8)
    ax.set_ylabel("Action index", fontsize=8)
    ax.set_xticks(np.arange(0, max_time + 1, max(1, max_time // 5)))
    y_ticks = np.arange(0, max_action + 1, max(1, (max_action + 1) // 10))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks + 1)
    ax.tick_params(axis="both", labelsize=7, length=2)
    ax.axhline(0.5, color="#7f2704", linewidth=0.8, linestyle="--", alpha=0.75)


def plot_policy_shift_heatmap(
    input_path: Path,
    output_dir: Path,
    *,
    dataset: str,
    method: str,
    rate: float,
    formats: Sequence[str],
) -> None:
    rows = _load_action_rows(
        input_path, dataset=dataset, method=method, rate=rate
    )
    if rows.is_empty():
        return

    initializers = [
        BASELINE_INITIALIZER,
        _initializer_for("mcar", rate),
        _initializer_for("mar", rate),
        _initializer_for("mnar_logistic", rate),
    ]
    rate_label = f"{rate:.0%}"
    titles = [
        "Cold",
        f"MCAR {rate_label}",
        f"MAR {rate_label}",
        f"MNAR logistic {rate_label}",
    ]
    max_action = cast("int", rows["action_performed"].max())
    max_time = cast("int", rows["n_selections_performed"].max())

    fig = plt.figure(figsize=(PLOT_WIDTH, 3.9))
    grid = fig.add_gridspec(
        1,
        len(initializers) + 1,
        width_ratios=[1, 1, 1, 1, 0.035],
        wspace=0.32,
    )
    axes = [
        fig.add_subplot(grid[0, index]) for index in range(len(initializers))
    ]
    colorbar_ax = fig.add_subplot(grid[0, -1])

    image = None
    for ax, initializer, title in zip(axes, initializers, titles, strict=True):
        subset = rows.filter(pl.col("initializer") == initializer)
        heatmap = _normalize_heatmap(subset, max_action, max_time)
        image = ax.imshow(
            heatmap,
            cmap="Blues",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        _format_action_axis(
            ax, title=title, max_action=max_action, max_time=max_time
        )

    if image is not None:
        colorbar = fig.colorbar(image, cax=colorbar_ax)
        colorbar.set_label("Selection frequency", fontsize=8)
        colorbar.ax.tick_params(labelsize=7, length=2)

    fig.subplots_adjust(left=0.055, right=0.975, top=0.82, bottom=0.16)

    for fmt in formats:
        path = output_dir / f"cube_nm_missing_policy_shift.{fmt}"
        fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = _load_filtered(args.input_path)
    metrics = collect_streaming(_compute_metrics_per_run(data))
    delta_rows = _compute_delta_rows(metrics)
    summary = _summarize_delta(delta_rows)

    metrics.write_csv(args.output_dir / "train_missing_metrics_per_run.csv")
    delta_rows.write_csv(args.output_dir / "train_missing_delta_per_run.csv")
    summary.write_csv(
        args.output_dir / "train_missing_diagnostics_summary.csv"
    )

    plot_delta_by_rate(summary, args.output_dir, args.formats)
    plot_baseline_percent_by_rate(summary, args.output_dir, args.formats)
    plot_family_relative_change_at_70(
        delta_rows, args.output_dir, args.formats
    )
    plot_delta_heatmap(summary, args.output_dir, args.formats)
    plot_policy_shift_heatmap(
        args.input_path,
        args.output_dir,
        dataset=args.heatmap_dataset,
        method=args.heatmap_method,
        rate=args.heatmap_rate,
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
