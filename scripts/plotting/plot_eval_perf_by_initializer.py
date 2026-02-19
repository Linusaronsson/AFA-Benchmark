from __future__ import annotations

import argparse
from pathlib import Path

import plotnine as p9
import polars as pl
from sklearn.metrics import accuracy_score, f1_score

from afabench.eval.plotting_config import (
    COLOR_PALETTE_NAME,
    DATASET_NAME_MAPPING,
    DATASETS_WITH_F_SCORE,
    METHOD_NAME_MAPPING,
)

INITIALIZER_NAME_MAPPING = {
    "missingness": "Missingness",
    "missingness_all_observed": "All observed",
}

VARIANT_NAME_MAPPING = {
    "marf": "MARF",
    "madt": "MADT",
    "magbt": "MAGBT",
    "malasso": "MALasso",
}


def _format_initializer_label(initializer: str) -> str:
    if initializer in INITIALIZER_NAME_MAPPING:
        return INITIALIZER_NAME_MAPPING[initializer]

    if initializer.startswith("missingness_all_observed_"):
        variant = initializer.removeprefix("missingness_all_observed_")
        variant_label = VARIANT_NAME_MAPPING.get(variant, variant.upper())
        return f"{variant_label} all observed"

    if initializer.startswith("missingness_"):
        variant = initializer.removeprefix("missingness_")
        variant_label = VARIANT_NAME_MAPPING.get(variant, variant.upper())
        return f"{variant_label} missingness"

    return initializer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot evaluation performance grouped by initializer, using "
            "already merged eval parquet from the workflow."
        )
    )
    parser.add_argument("input_path", type=Path, help="Input parquet path.")
    parser.add_argument("output_dir", type=Path, help="Output directory.")
    parser.add_argument(
        "--budget-mode",
        type=str,
        default="hard",
        choices=["hard", "soft", "all"],
        help="Which budget regime to include.",
    )
    parser.add_argument(
        "--keep-largest-hard-budget",
        action="store_true",
        help=(
            "When budget-mode=hard, keep only the largest hard budget per "
            "dataset."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional list of methods to keep.",
    )
    return parser.parse_args()


def _compute_metrics_per_run(data: pl.DataFrame) -> pl.DataFrame:
    group_cols = [
        "afa_method",
        "dataset",
        "initializer",
        "train_seed",
        "eval_seed",
        "eval_hard_budget",
        "train_hard_budget",
        "train_soft_budget_param",
        "eval_soft_budget_param",
    ]

    return data.group_by(*group_cols).map_groups(
        lambda group_df: pl.DataFrame(
            {
                "afa_method": [group_df["afa_method"].first()],
                "dataset": [group_df["dataset"].first()],
                "initializer": [group_df["initializer"].first()],
                "train_seed": [group_df["train_seed"].first()],
                "eval_seed": [group_df["eval_seed"].first()],
                "eval_hard_budget": [group_df["eval_hard_budget"].first()],
                "train_hard_budget": [group_df["train_hard_budget"].first()],
                "train_soft_budget_param": [
                    group_df["train_soft_budget_param"].first()
                ],
                "eval_soft_budget_param": [
                    group_df["eval_soft_budget_param"].first()
                ],
                "accuracy": [
                    accuracy_score(
                        group_df["true_class"], group_df["predicted_class"]
                    )
                ],
                "f_score": [
                    f1_score(
                        group_df["true_class"],
                        group_df["predicted_class"],
                        average="macro",
                    )
                ],
            },
            schema={
                "afa_method": pl.String,
                "dataset": pl.String,
                "initializer": pl.String,
                "train_seed": pl.UInt64,
                "eval_seed": pl.UInt64,
                "eval_hard_budget": pl.Float64,
                "train_hard_budget": pl.Float64,
                "train_soft_budget_param": pl.Float64,
                "eval_soft_budget_param": pl.Float64,
                "accuracy": pl.Float64,
                "f_score": pl.Float64,
            },
        )
    )


def _aggregate_across_seeds(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        metric=pl.when(pl.col("dataset").is_in(DATASETS_WITH_F_SCORE))
        .then(pl.col("f_score"))
        .otherwise(pl.col("accuracy"))
    )
    return (
        data.group_by("dataset", "afa_method", "initializer")
        .agg(
            mean_metric=pl.col("metric").mean(),
            std_metric=pl.col("metric").std(),
            n_runs=pl.len(),
        )
        .with_columns(std_metric=pl.col("std_metric").fill_null(0.0))
        .with_columns(
            low_metric=pl.col("mean_metric") - pl.col("std_metric"),
            high_metric=pl.col("mean_metric") + pl.col("std_metric"),
        )
    )


def _filter_budget_mode(
    data: pl.DataFrame,
    mode: str,
    keep_largest_hard_budget: bool,
) -> pl.DataFrame:
    if mode == "hard":
        data = data.filter(pl.col("eval_hard_budget").is_not_null())
        if keep_largest_hard_budget:
            data = data.filter(
                pl.col("eval_hard_budget")
                == pl.col("eval_hard_budget").max().over("dataset")
            )
        return data
    if mode == "soft":
        return data.filter(
            pl.col("train_soft_budget_param").is_not_null()
            | pl.col("eval_soft_budget_param").is_not_null()
        )
    return data


def _make_plot(summary_data: pl.DataFrame) -> p9.ggplot:
    plot_data = summary_data.with_columns(
        dataset=pl.col("dataset").replace(DATASET_NAME_MAPPING),
        afa_method=pl.col("afa_method").replace(METHOD_NAME_MAPPING),
        initializer=pl.col("initializer").map_elements(
            _format_initializer_label,
            return_dtype=pl.String,
        ),
    )

    n_datasets = max(plot_data["dataset"].n_unique(), 1)
    n_rows = (n_datasets + 1) // 2
    figure_height = 2.8 * n_rows

    return (
        p9.ggplot(
            plot_data,
            p9.aes(
                x="initializer",
                y="mean_metric",
                color="initializer",
                shape="afa_method",
            ),
        )
        + p9.geom_point(size=2.8)
        + p9.geom_errorbar(
            p9.aes(ymin="low_metric", ymax="high_metric"),
            width=0.08,
        )
        + p9.facet_wrap("dataset", scales="free_y", ncol=2)
        + p9.scale_color_brewer(type="qual", palette=COLOR_PALETTE_NAME)
        + p9.labs(
            x="Initializer",
            y="Metric",
            color="Initializer",
            shape="Policy",
        )
        + p9.theme(figure_size=(11.0, figure_height))
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pl.read_parquet(args.input_path)
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
        "train_soft_budget_param",
        "eval_soft_budget_param",
    }
    missing_cols = required_cols.difference(data.columns)
    if missing_cols:
        msg = f"Input parquet missing required columns: {sorted(missing_cols)}"
        raise ValueError(msg)

    if args.methods:
        data = data.filter(pl.col("afa_method").is_in(args.methods))

    data = data.filter(
        (pl.col("action_performed") == 0)
        & pl.col("predicted_class").is_not_null()
    )
    data = _filter_budget_mode(
        data, args.budget_mode, args.keep_largest_hard_budget
    )
    if data.is_empty():
        msg = "No rows left after filtering; cannot generate plot."
        raise ValueError(msg)

    metrics_per_run = _compute_metrics_per_run(data)
    summary_data = _aggregate_across_seeds(metrics_per_run)
    if summary_data.is_empty():
        msg = "No grouped rows produced; cannot generate plot."
        raise ValueError(msg)

    plot = _make_plot(summary_data)
    plot.save(
        args.output_dir / "initializer_comparison.pdf",
        width=11.0,
        height=max(3.0, 2.8 * ((summary_data["dataset"].n_unique() + 1) // 2)),
    )
    summary_data.write_csv(args.output_dir / "initializer_comparison.csv")


if __name__ == "__main__":
    main()
