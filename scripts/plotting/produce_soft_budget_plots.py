from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_cartesian,
    facet_wrap,
    geom_errorbar,
    geom_errorbarh,
    geom_line,
    geom_point,
    ggplot,
    labs,
    theme_bw,
    theme_void,
)

# Datasets that use F1 instead of accuracy
F1_DATASETS = {"physionet"}

# Method name mapping
METHOD_NAME_MAPPING = {
    "aaco": "AACO",
    "covert2023": "GDFS",
    "gadgil2023": "DIME",
    "kachuee2019": "OL-MFRL",
    "ma2018": "EDDI",
    "shim2018": "JAFA-MFRL",
    "zannone2019": "ODIN-MFRL",
    "random_dummy": "Random",
}


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read evaluation CSV with proper column types."""
    df = pd.read_csv(path)

    # Convert to appropriate types
    int_cols = [
        "features_chosen",
        "predicted_label_external",
        "true_label",
        "predicted_label_builtin",
        "training_seed",
        "dataset_split",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    float_cols = ["acquisition_cost", "cost_parameter"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical_cols = ["method", "dataset"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def tidy_df(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy the dataframe by renaming methods and pivoting predictions."""
    df = df.copy()

    # Rename methods
    df["method"] = (
        df["method"]
        .map(lambda x: METHOD_NAME_MAPPING.get(x, x))
        .astype("category")
    )

    # Pivot predictions from wide to long format
    id_vars = [
        col
        for col in df.columns
        if col not in ["predicted_label_external", "predicted_label_builtin"]
    ]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=["predicted_label_external", "predicted_label_builtin"],
        var_name="prediction_type",
        value_name="predicted_label",
    )

    # Clean up prediction type names
    df_long["prediction_type"] = df_long["prediction_type"].str.replace(
        "predicted_label_", ""
    )

    # Remove rows where prediction is NA
    df_long = df_long.loc[df_long["predicted_label"].notna()].copy()

    return df_long


def summarize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize metrics per group."""
    df = df.copy()

    # Calculate binary classification metrics
    df["correct"] = (df["predicted_label"] == df["true_label"]).astype(int)
    df["tp"] = ((df["predicted_label"] == 1) & (df["true_label"] == 1)).astype(
        int
    )
    df["fp"] = ((df["predicted_label"] == 1) & (df["true_label"] == 0)).astype(
        int
    )
    df["tn"] = ((df["predicted_label"] == 0) & (df["true_label"] == 0)).astype(
        int
    )
    df["fn"] = ((df["predicted_label"] == 0) & (df["true_label"] == 1)).astype(
        int
    )

    # Group and aggregate
    group_cols = [
        "method",
        "prediction_type",
        "dataset",
        "dataset_split",
        "cost_parameter",
        "training_seed",
    ]

    agg_df = (
        df.groupby(group_cols, observed=True)
        .agg(
            accuracy=("correct", "mean"),
            avg_features_chosen=("features_chosen", "mean"),
            avg_acquisition_cost=("acquisition_cost", "mean"),
            tp=("tp", "sum"),
            fp=("fp", "sum"),
            tn=("tn", "sum"),
            fn=("fn", "sum"),
        )
        .reset_index()
    )

    # Calculate precision, recall, F1
    agg_df["precision"] = np.where(
        agg_df["tp"] + agg_df["fp"] == 0,
        0,
        agg_df["tp"] / (agg_df["tp"] + agg_df["fp"]),
    )
    agg_df["recall"] = np.where(
        agg_df["tp"] + agg_df["fn"] == 0,
        0,
        agg_df["tp"] / (agg_df["tp"] + agg_df["fn"]),
    )
    agg_df["f1"] = np.where(
        agg_df["precision"] + agg_df["recall"] == 0,
        0,
        2
        * (agg_df["precision"] * agg_df["recall"])
        / (agg_df["precision"] + agg_df["recall"]),
    )

    # Determine metric type based on dataset
    agg_df["metric_type"] = agg_df["dataset"].apply(
        lambda x: "f1" if x in F1_DATASETS else "accuracy"
    )
    agg_df["metric_value"] = np.where(
        agg_df["metric_type"] == "f1",
        agg_df["f1"],
        agg_df["accuracy"],
    )

    # Summarize across dataset splits and training seeds
    summary_cols = [
        "method",
        "prediction_type",
        "dataset",
        "cost_parameter",
        "metric_type",
    ]
    summary_df = (
        agg_df.groupby(summary_cols, observed=True)
        .agg(
            avg_metric=("metric_value", "mean"),
            sd_metric=("metric_value", "std"),
            mean_avg_features_chosen=("avg_features_chosen", "mean"),
            sd_avg_features_chosen=("avg_features_chosen", "std"),
            mean_avg_acquisition_cost=("avg_acquisition_cost", "mean"),
            sd_avg_acquisition_cost=("avg_acquisition_cost", "std"),
        )
        .reset_index()
    )

    # Fill NaN standard deviations with 0
    summary_df = summary_df.fillna(0)

    return summary_df


def empty_plot(title_text: str) -> ggplot:
    """Create an empty plot with a message."""
    return (
        ggplot()
        + theme_void()
        + labs(title=title_text, subtitle="No data to plot")
    )


def get_soft_budget_plot(df: pd.DataFrame, classifier_type: str) -> ggplot:
    """Create soft budget plot for a specific classifier type."""
    # Filter to only include the specific classifier type
    df = df.loc[df["prediction_type"] == classifier_type].copy()

    if len(df) == 0:
        return empty_plot(
            f"Metric vs avg. acquisition cost ({classifier_type})"
        )

    # Filter out datasets that only have a single method
    method_counts = df.groupby(
        "dataset",
        observed=True,
        as_index=False,
    )["method"].nunique()
    method_counts["method_count"] = method_counts["method"]
    method_counts = method_counts.drop(columns=["method"])
    valid_datasets = method_counts.loc[
        method_counts["method_count"] > 1,
        "dataset",
    ].to_list()
    df = df.loc[df["dataset"].isin(valid_datasets)].copy()

    if len(df) == 0:
        return empty_plot(
            f"Metric vs avg. acquisition cost ({classifier_type})"
        )

    plot = (
        ggplot(
            df,
            aes(
                x="mean_avg_acquisition_cost",
                y="avg_metric",
                color="method",
            ),
        )
        + geom_point()
        + geom_line()
        + geom_errorbar(
            aes(
                ymin="avg_metric - sd_metric",
                ymax="avg_metric + sd_metric",
            ),
            width=0,
        )
        + geom_errorbarh(
            aes(
                xmin="mean_avg_acquisition_cost - sd_avg_acquisition_cost",
                xmax="mean_avg_acquisition_cost + sd_avg_acquisition_cost",
            ),
            height=0,
        )
        + facet_wrap("~dataset", scales="free")
        + coord_cartesian(ylim=(0, 1))
        + labs(
            title=f"Metric vs avg. acquisition cost ({classifier_type})",
            x="Avg. acquisition cost",
            y="Metric",
            color="Method",
        )
        + theme_bw()
    )

    return plot


def get_params_plot(df: pd.DataFrame) -> ggplot:
    """Create cost parameter vs features chosen plot."""
    if len(df) == 0:
        return empty_plot("Cost parameter vs features chosen")

    # Create a combined facet label
    df = df.copy()
    df["facet_label"] = (
        df["method"].astype(str) + " / " + df["dataset"].astype(str)
    )

    plot = (
        ggplot(
            df,
            aes(x="mean_avg_features_chosen", y="cost_parameter"),
        )
        + geom_point()
        + facet_wrap("~facet_label", scales="free")
        + labs(
            title="Cost parameter vs features chosen",
            x="Mean avg. features chosen",
            y="Cost parameter",
        )
        + theme_bw()
    )

    return plot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce soft budget evaluation plots"
    )
    parser.add_argument(
        "eval_results",
        type=Path,
        help="Path to evaluation results CSV",
    )
    parser.add_argument(
        "output_plot1",
        type=Path,
        help="Output path for builtin classifier plot",
    )
    parser.add_argument(
        "output_plot2",
        type=Path,
        help="Output path for external classifier plot",
    )
    parser.add_argument(
        "cost_params_plot",
        type=Path,
        help="Output path for cost parameters plot",
    )

    args = parser.parse_args()

    # Read and process data
    df = read_csv_safe(args.eval_results)
    tidied_df = tidy_df(df)
    summarized_df = summarize_df(tidied_df)

    # Check which classifier types are available
    has_builtin = "builtin" in summarized_df["prediction_type"].to_numpy()
    has_external = "external" in summarized_df["prediction_type"].to_numpy()

    # Create plots
    if has_builtin:
        soft_budget_plot_builtin = get_soft_budget_plot(
            summarized_df, "builtin"
        )
    else:
        soft_budget_plot_builtin = empty_plot(
            "Metric vs avg. acquisition cost (builtin)"
        )

    if has_external:
        soft_budget_plot_external = get_soft_budget_plot(
            summarized_df, "external"
        )
    else:
        soft_budget_plot_external = empty_plot(
            "Metric vs avg. acquisition cost (external)"
        )

    params_plot = get_params_plot(summarized_df)

    # Save plots
    args.output_plot1.parent.mkdir(parents=True, exist_ok=True)
    soft_budget_plot_builtin.save(
        args.output_plot1, width=10, height=6, dpi=300, verbose=False
    )
    soft_budget_plot_external.save(
        args.output_plot2, width=10, height=6, dpi=300, verbose=False
    )
    params_plot.save(
        args.cost_params_plot, width=10, height=6, dpi=300, verbose=False
    )

    print(
        f"Saved plots to {args.output_plot1}, "
        f"{args.output_plot2}, "
        f"{args.cost_params_plot}"
    )


if __name__ == "__main__":
    main()
