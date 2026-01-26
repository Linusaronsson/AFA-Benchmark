from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    facet_wrap,
    geom_errorbar,
    geom_errorbarh,
    geom_line,
    geom_ribbon,
    ggplot,
    labs,
    scale_color_discrete,
    scale_fill_discrete,
    theme,
)
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

METHOD_NAME_MAPPING = {
    "random_dummy": "Random dummy",
    "sequential_dummy": "Sequential dummy",
    "shim2018": "Shim 2018",
    "eddi": "EDDI",
    "dime": "DIME",
    "zannone2019": "Zannone 2019",
    "aaco": "AACO",
    "aaco_nn": "AACO+NN",
}

DATASET_NAME_MAPPING = {
    "cube": "Cube",
    "afa_context": "AFAContext",
    "synthetic_mnist": "Synthetic MNIST",
    "cube_without_noise": "Cube (noiseless)",
    "afa_context_without_noise": "AFAContext (noiseless)",
    "synthetic_mnist_without_noise": "Synthetic MNIST (noiseless)",
    "MNIST": "MNIST",
}

DATASET_METRIC_MAPPING = {
    "cube": "accuracy",
    "afa_context": "f1",
    "synthetic_mnist": "accuracy",
    "cube_without_noise": "accuracy",
    "afa_context_without_noise": "f1",
    "synthetic_mnist_without_noise": "accuracy",
    "MNIST": "f1",
}

METRIC_NAME_MAPPING = {
    "accuracy": "Accuracy",
    "f1": "F1",
    "kappa": "Cohen's Kappa",
}

DTYPE_SPEC = {
    "afa_method": "category",
    "dataset": "category",
    "predicted_class": "category",
    "true_class": "category",
    "train_seed": "int64",
    "eval_seed": "int64",
    "accumulated_cost": "float64",
    "eval_hard_budget": "float64",
    "train_soft_budget_param": "float64",
}


def dataset_with_metric_labeller(dataset: str) -> str:
    """Create facet label with dataset name and metric."""
    dataset_name = DATASET_NAME_MAPPING.get(dataset, dataset)
    metric_code = DATASET_METRIC_MAPPING.get(dataset, "accuracy")
    metric_name = METRIC_NAME_MAPPING.get(metric_code, metric_code)
    return f"{dataset_name} ({metric_name})"


def apply_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply display labels to method codes."""
    df = df.copy()
    df["afa_method"] = (
        df["afa_method"]
        .map(lambda x: METHOD_NAME_MAPPING.get(x, x))
        .astype("category")
    )
    return df


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read evaluation CSV with proper column types."""
    df = pd.read_csv(path, dtype=DTYPE_SPEC, na_values=["null"])  # pyright: ignore[reportCallIssue, reportArgumentType]
    return df


def compute_classification_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute classification metrics for a group."""
    y_true = group["true_class"].to_numpy()
    y_pred = group["predicted_class"].to_numpy()

    if pd.isna(y_pred).any() or pd.isna(y_true).any():
        return pd.Series({"accuracy": np.nan, "f1": np.nan, "kappa": np.nan})

    acc = accuracy_score(y_true, y_pred)
    # Macro average F1 including all classes (standard approach)
    # sklearn's f1_score with average="macro" automatically includes all classes
    f1 = f1_score(y_true, y_pred, average="macro", zero_division="warn")
    kappa = cohen_kappa_score(y_true, y_pred)

    return pd.Series({"accuracy": acc, "f1": f1, "kappa": kappa})


def create_hard_budget_plot(
    df: pd.DataFrame,
    y_label: str,
    *,
    use_dataset_metric_labeller: bool = True,
) -> ggplot:
    """Create hard budget plot with lines and ribbon."""
    # Pre-compute ribbon bounds and facet label
    df = df.copy()
    df["y_min"] = df["estimate_mean"] - df["estimate_sd"]
    df["y_max"] = df["estimate_mean"] + df["estimate_sd"]

    if use_dataset_metric_labeller:
        df["facet_label"] = df["dataset"].apply(dataset_with_metric_labeller)
    else:
        df["facet_label"] = df["dataset"].apply(
            lambda x: DATASET_NAME_MAPPING.get(x, x)
        )

    plot = (
        ggplot(
            df,
            aes(
                x="eval_hard_budget",
                y="estimate_mean",
                color="afa_method",
                fill="afa_method",
            ),
        )
        + geom_line()
        + geom_ribbon(
            aes(
                ymin="y_min",
                ymax="y_max",
            ),
            alpha=0.2,
            color=None,
        )
        + facet_wrap("~facet_label", scales="free")
        + labs(x="Selection budget", y=y_label)
        + scale_color_discrete(name="AFA method")
        + scale_fill_discrete(name="AFA method")
        + theme(legend_position="bottom")
    )
    return plot


def create_soft_budget_plot(
    df: pd.DataFrame,
    y_label: str,
    *,
    use_dataset_metric_labeller: bool = True,
) -> ggplot:
    """Create soft budget plot with error bars."""
    # Pre-compute error bar bounds and facet label
    df = df.copy()
    df["y_min"] = df["estimate_mean"] - df["estimate_sd"]
    df["y_max"] = df["estimate_mean"] + df["estimate_sd"]
    df["x_min"] = df["accumulated_cost_mean"] - df["accumulated_cost_sd"]
    df["x_max"] = df["accumulated_cost_mean"] + df["accumulated_cost_sd"]

    if use_dataset_metric_labeller:
        df["facet_label"] = df["dataset"].apply(dataset_with_metric_labeller)
    else:
        df["facet_label"] = df["dataset"].apply(
            lambda x: DATASET_NAME_MAPPING.get(x, x)
        )

    plot = (
        ggplot(
            df,
            aes(
                x="accumulated_cost_mean",
                y="estimate_mean",
                color="afa_method",
                fill="afa_method",
            ),
        )
        + geom_line()
        + geom_errorbar(
            aes(ymin="y_min", ymax="y_max"),
            alpha=0.5,
            width=0,
        )
        + geom_errorbarh(
            aes(xmin="x_min", xmax="x_max"),
            alpha=0.5,
            height=0,
        )
        + facet_wrap("~facet_label", scales="free")
        + labs(x="Accumulated cost", y=y_label)
        + scale_color_discrete(name="AFA method")
        + scale_fill_discrete(name="AFA method")
        + theme(legend_position="bottom")
    )

    return plot


def metrics_grouped_by_param(df: pd.DataFrame, param: str) -> pd.DataFrame:
    group_cols = [
        "afa_method",
        "dataset",
        "train_seed",
        "eval_seed",
        param,
    ]
    metrics_list = []

    for name, group in df.groupby(group_cols, observed=True):
        assert isinstance(group, pd.DataFrame)
        group_name = name if isinstance(name, tuple) else (name,)
        metrics = compute_classification_metrics(group)
        row = dict(zip(group_cols, group_name, strict=True))
        row.update(metrics.to_dict())
        # Add accumulated_cost from the group
        row["accumulated_cost"] = group["accumulated_cost"].mean()  # pyright: ignore[reportArgumentType]
        metrics_list.append(row)

    if not metrics_list:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics_list)

    # Everything in group_cols, except for seeds
    summary_cols = [
        col for col in group_cols if col not in ("train_seed", "eval_seed")
    ]
    summary = (
        metrics_df.groupby(summary_cols, observed=True)
        .agg(
            accumulated_cost_mean=("accumulated_cost", "mean"),
            accumulated_cost_sd=("accumulated_cost", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            f1_mean=("f1", "mean"),
            f1_sd=("f1", "std"),
            kappa_mean=("kappa", "mean"),
            kappa_sd=("kappa", "std"),
        )
        .reset_index()
    )

    return summary


def process_hard_budget_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process data for hard budget plots."""
    df_hard = df.loc[
        df["eval_hard_budget"].notna() & df["train_soft_budget_param"].isna()
    ].copy()
    df_hard = metrics_grouped_by_param(df_hard, param="eval_hard_budget")
    return df_hard


def process_soft_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Process data for soft budget plots."""
    df_soft = df.loc[
        df["eval_hard_budget"].isna() & df["train_soft_budget_param"].notna()
    ].copy()
    df_soft = metrics_grouped_by_param(
        df_soft, param="train_soft_budget_param"
    )
    return df_soft


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from merged evaluation results.",
    )
    parser.add_argument(
        "input_folder",
        type=Path,
        help="Input CSV file with evaluation results.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Output folder where plots will be saved",
    )
    return parser.parse_args()


def get_metric_estimate(row: pd.Series, metric: str) -> float:
    """Extract metric estimate from row based on dataset-specific metric."""
    dataset = str(row["dataset"])
    metric_code = DATASET_METRIC_MAPPING.get(dataset, "accuracy")
    return float(row[f"{metric_code}_{metric}"])


def add_estimate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add estimate_mean and estimate_sd columns using dataset-specific metrics."""
    df = df.copy()
    df["estimate_mean"] = df.apply(
        lambda row: get_metric_estimate(row, "mean"), axis=1
    )
    df["estimate_sd"] = df.apply(
        lambda row: get_metric_estimate(row, "sd"), axis=1
    )
    return df


def save_plot(plot: ggplot, output_path: Path, dpi: int = 150) -> None:
    """Save a ggplot to file."""
    plot.save(output_path, dpi=dpi, verbose=False)


def produce_hard_budget_plots(df: pd.DataFrame, output_folder: Path) -> None:
    df_hard_budget = process_hard_budget_df(df)

    # Add estimate columns using dataset-specific metrics
    builtin_hard = add_estimate_columns(df_hard_budget)

    # Main metric plot
    hard_budget_plot = create_hard_budget_plot(builtin_hard, "Metric")
    save_plot(hard_budget_plot, output_folder / "hard_budget.png")


def produce_soft_budget_plots(df: pd.DataFrame, output_folder: Path) -> None:
    """Produce soft budget plots."""
    df_soft_budget = process_soft_budget(df)

    if len(df_soft_budget) == 0:
        print("No soft budget data found in the evaluation results.")
        return

    # Add estimate columns using dataset-specific metrics
    builtin_soft = add_estimate_columns(df_soft_budget)

    # Main metric plot
    soft_budget_plot = create_soft_budget_plot(builtin_soft, "Metric")
    save_plot(soft_budget_plot, output_folder / "soft_budget.png")


def only_last_step_in_episode(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    dfc = dfc[dfc["action_performed"] == 0]
    assert isinstance(dfc, pd.DataFrame)
    return dfc


def main() -> None:
    args = parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    df = read_csv_safe(args.input_folder)
    df = only_last_step_in_episode(df)

    produce_hard_budget_plots(df, args.output_folder)
    produce_soft_budget_plots(df, args.output_folder)


if __name__ == "__main__":
    main()
