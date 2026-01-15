"""
Generate evaluation plots from merged evaluation results.

Replaces scripts/plotting/plot_eval.R with pure Python using plotnine.

Usage:
    python plot_eval.py [eval_results.csv] output_folder

If no input CSV is provided, generates dummy data for testing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    facet_wrap,
    geom_errorbar,
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


def dataset_with_metric_labeller(dataset: str) -> str:
    """Create facet label with dataset name and metric."""
    dataset_name = DATASET_NAME_MAPPING.get(dataset, dataset)
    metric_code = DATASET_METRIC_MAPPING.get(dataset, "accuracy")
    metric_name = METRIC_NAME_MAPPING.get(metric_code, metric_code)
    return f"{dataset_name} ({metric_name})"


def apply_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Apply display labels to method codes."""
    if "afa_method" not in df.columns:
        return df
    df = df.copy()
    df["afa_method"] = (
        df["afa_method"]
        .map(lambda x: METHOD_NAME_MAPPING.get(x, x))
        .astype("category")
    )
    return df


def read_csv_safe(path: Path) -> pd.DataFrame:
    """Read evaluation CSV with proper column types."""
    df = pd.read_csv(path)

    categorical_cols = ["afa_method", "classifier", "dataset"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    int_cols = [
        "selections_performed",
        "predicted_class",
        "true_class",
        "train_seed",
        "eval_seed",
        "hard_budget",
    ]
    for col in int_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            assert isinstance(series, pd.Series)
            df[col] = series.astype("Int64")

    if "soft_budget_param" in df.columns:
        series = pd.to_numeric(df["soft_budget_param"], errors="coerce")
        assert isinstance(series, pd.Series)
        df["soft_budget_param"] = series

    return apply_method_labels(df)


def generate_dummy_data(n: int = 10000) -> pd.DataFrame:
    """Generate dummy data for testing."""
    rng = np.random.default_rng(42)
    methods = ["random_dummy", "sequential_dummy", "aaco"]
    classifiers = ["builtin", "external"]
    datasets = ["cube", "afa_context", "synthetic_mnist"]
    train_seeds = [1, 2]
    eval_seeds = [1, 2]
    hard_budgets = [5, 10, 15]
    n_classes = 5

    df = pd.DataFrame(
        {
            "afa_method": rng.choice(methods, n),
            "classifier": rng.choice(classifiers, n),
            "dataset": rng.choice(datasets, n),
            "selections_performed": rng.integers(1, 16, n),
            "predicted_class": rng.integers(0, n_classes, n),
            "true_class": rng.integers(0, n_classes, n),
            "train_seed": rng.choice(train_seeds, n),
            "eval_seed": rng.choice(eval_seeds, n),
            "hard_budget": rng.choice(hard_budgets, n),
            "soft_budget_param": rng.choice([np.nan, 0.1, 1.0], n),
        }
    )

    for col in ["afa_method", "classifier", "dataset"]:
        df[col] = df[col].astype("category")

    return apply_method_labels(df)


def compute_metrics(group: pd.DataFrame) -> pd.Series:
    """Compute classification metrics for a group."""
    y_true = group["true_class"].to_numpy()
    y_pred = group["predicted_class"].to_numpy()

    if pd.isna(y_pred).any() or pd.isna(y_true).any():
        return pd.Series({"accuracy": np.nan, "f1": np.nan, "kappa": np.nan})

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)

    return pd.Series({"accuracy": acc, "f1": f1, "kappa": kappa})


def create_hard_budget_plot(
    df: pd.DataFrame,
    y_label: str,
    *,
    use_dataset_metric_labeller: bool = True,
) -> ggplot:
    """Create hard budget plot with lines and ribbon."""
    if use_dataset_metric_labeller:
        df = df.copy()
        df["facet_label"] = df["dataset"].apply(dataset_with_metric_labeller)
    else:
        df = df.copy()
        df["facet_label"] = df["dataset"].apply(
            lambda x: DATASET_NAME_MAPPING.get(x, x)
        )

    plot = (
        ggplot(
            df,
            aes(
                x="hard_budget",
                y="estimate_mean",
                color="afa_method",
                fill="afa_method",
            ),
        )
        + geom_line()
        + geom_ribbon(
            aes(
                ymin="estimate_mean - estimate_sd",
                ymax="estimate_mean + estimate_sd",
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
    if use_dataset_metric_labeller:
        df = df.copy()
        df["facet_label"] = df["dataset"].apply(dataset_with_metric_labeller)
    else:
        df = df.copy()
        df["facet_label"] = df["dataset"].apply(
            lambda x: DATASET_NAME_MAPPING.get(x, x)
        )

    plot = (
        ggplot(
            df,
            aes(
                x="selections_performed_mean",
                y="estimate_mean",
                color="afa_method",
                fill="afa_method",
            ),
        )
        + geom_line()
        + geom_errorbar(
            aes(
                ymin="estimate_mean - estimate_sd",
                ymax="estimate_mean + estimate_sd",
            ),
            alpha=0.5,
        )
        + geom_errorbar(
            aes(
                xmin="selections_performed_mean - selections_performed_sd",
                xmax="selections_performed_mean + selections_performed_sd",
            ),
            alpha=0.5,
        )
        + facet_wrap("~facet_label", scales="free")
        + labs(x="Selection budget", y=y_label)
        + scale_color_discrete(name="AFA method")
        + scale_fill_discrete(name="AFA method")
        + theme(legend_position="bottom")
    )
    return plot


def process_hard_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Process data for hard budget plots."""
    df_hard = df.loc[
        df["hard_budget"].notna() & df["soft_budget_param"].isna()
    ].copy()

    if len(df_hard) == 0:
        return pd.DataFrame()

    df_hard = df_hard.loc[
        df_hard["selections_performed"] == df_hard["hard_budget"]
    ]

    if len(df_hard) == 0:
        return pd.DataFrame()

    group_cols = [
        "afa_method",
        "classifier",
        "dataset",
        "train_seed",
        "eval_seed",
        "hard_budget",
    ]
    metrics_list = []

    for name, group in df_hard.groupby(group_cols, observed=True):
        assert isinstance(group, pd.DataFrame)
        group_name = name if isinstance(name, tuple) else (name,)
        metrics = compute_metrics(group)
        row = dict(zip(group_cols, group_name, strict=True))
        row.update(metrics.to_dict())
        metrics_list.append(row)

    if not metrics_list:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics_list)

    summary_cols = ["afa_method", "classifier", "dataset", "hard_budget"]
    summary = (
        metrics_df.groupby(summary_cols, observed=True)
        .agg(
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


def process_soft_budget(df: pd.DataFrame) -> pd.DataFrame:
    """Process data for soft budget plots."""
    df_soft = df.loc[
        df["hard_budget"].isna() & df["soft_budget_param"].notna()
    ].copy()

    if len(df_soft) == 0:
        return pd.DataFrame()

    group_cols = [
        "afa_method",
        "classifier",
        "dataset",
        "train_seed",
        "eval_seed",
        "soft_budget_param",
    ]

    selections_df = (
        df_soft.groupby(group_cols, observed=True)
        .agg(avg_selections_performed=("selections_performed", "mean"))
        .reset_index()
    )

    metrics_list = []
    for name, group in df_soft.groupby(group_cols, observed=True):
        assert isinstance(group, pd.DataFrame)
        group_name = name if isinstance(name, tuple) else (name,)
        metrics = compute_metrics(group)
        row = dict(zip(group_cols, group_name, strict=True))
        row.update(metrics.to_dict())
        metrics_list.append(row)

    if not metrics_list:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics_list)

    combined = metrics_df.merge(selections_df, on=group_cols)

    summary_cols = ["afa_method", "classifier", "dataset", "soft_budget_param"]
    summary = (
        combined.groupby(summary_cols, observed=True)
        .agg(
            selections_performed_mean=("avg_selections_performed", "mean"),
            selections_performed_sd=("avg_selections_performed", "std"),
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


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Run the main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation plots from merged results"
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="[input_csv] output_folder",
    )

    args = parser.parse_args()

    if len(args.args) == 0:
        print("Proceeding with plotting using dummy data...")
        df = generate_dummy_data(n=10000)
        output_path = None
    elif len(args.args) == 1:
        print("Proceeding with plotting using dummy data...")
        df = generate_dummy_data(n=10000)
        output_path = Path(args.args[0])
    elif len(args.args) == 2:
        df = read_csv_safe(Path(args.args[0]))
        output_path = Path(args.args[1])
    else:
        print("Usage: python plot_eval.py [eval_results.csv] output_folder")
        sys.exit(1)

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    df_hard_budget = process_hard_budget(df)

    if len(df_hard_budget) > 0:
        builtin_hard = df_hard_budget.loc[
            df_hard_budget["classifier"] == "builtin"
        ].copy()
        assert isinstance(builtin_hard, pd.DataFrame)

        if len(builtin_hard) > 0:

            def get_estimate(row: pd.Series) -> float:
                dataset = str(row["dataset"])
                metric = DATASET_METRIC_MAPPING.get(dataset, "accuracy")
                return float(row[f"{metric}_mean"])

            def get_estimate_sd(row: pd.Series) -> float:
                dataset = str(row["dataset"])
                metric = DATASET_METRIC_MAPPING.get(dataset, "accuracy")
                return float(row[f"{metric}_sd"])

            builtin_hard["estimate_mean"] = builtin_hard.apply(
                get_estimate, axis=1
            )
            builtin_hard["estimate_sd"] = builtin_hard.apply(
                get_estimate_sd, axis=1
            )

            hard_budget_plot = create_hard_budget_plot(builtin_hard, "Metric")
            if output_path:
                hard_budget_plot.save(
                    output_path / "hard_budget.png", dpi=150, verbose=False
                )
            else:
                print(hard_budget_plot)

            builtin_hard_kappa = builtin_hard.copy()
            builtin_hard_kappa["estimate_mean"] = builtin_hard_kappa[
                "kappa_mean"
            ]
            builtin_hard_kappa["estimate_sd"] = builtin_hard_kappa["kappa_sd"]
            hard_budget_kappa_plot = create_hard_budget_plot(
                builtin_hard_kappa,
                "Cohen's Kappa",
                use_dataset_metric_labeller=False,
            )
            if output_path:
                hard_budget_kappa_plot.save(
                    output_path / "hard_budget_kappa.png",
                    dpi=150,
                    verbose=False,
                )
            else:
                print(hard_budget_kappa_plot)

    df_soft_budget = process_soft_budget(df)

    if len(df_soft_budget) > 0:
        builtin_soft = df_soft_budget.loc[
            df_soft_budget["classifier"] == "builtin"
        ].copy()
        assert isinstance(builtin_soft, pd.DataFrame)

        if len(builtin_soft) > 0:

            def get_estimate(row: pd.Series) -> float:
                dataset = str(row["dataset"])
                metric = DATASET_METRIC_MAPPING.get(dataset, "accuracy")
                return float(row[f"{metric}_mean"])

            def get_estimate_sd(row: pd.Series) -> float:
                dataset = str(row["dataset"])
                metric = DATASET_METRIC_MAPPING.get(dataset, "accuracy")
                return float(row[f"{metric}_sd"])

            builtin_soft["estimate_mean"] = builtin_soft.apply(
                get_estimate, axis=1
            )
            builtin_soft["estimate_sd"] = builtin_soft.apply(
                get_estimate_sd, axis=1
            )

            soft_budget_plot = create_soft_budget_plot(builtin_soft, "Metric")
            if output_path:
                soft_budget_plot.save(
                    output_path / "soft_budget.png", dpi=150, verbose=False
                )
            else:
                print(soft_budget_plot)

            builtin_soft_kappa = builtin_soft.copy()
            builtin_soft_kappa["estimate_mean"] = builtin_soft_kappa[
                "kappa_mean"
            ]
            builtin_soft_kappa["estimate_sd"] = builtin_soft_kappa["kappa_sd"]
            soft_budget_kappa_plot = create_soft_budget_plot(
                builtin_soft_kappa,
                "Cohen's Kappa",
                use_dataset_metric_labeller=False,
            )
            if output_path:
                soft_budget_kappa_plot.save(
                    output_path / "soft_budget_kappa.png",
                    dpi=150,
                    verbose=False,
                )
            else:
                print(soft_budget_kappa_plot)

    if output_path:
        print(f"Plots saved to {output_path}")


if __name__ == "__main__":
    main()
