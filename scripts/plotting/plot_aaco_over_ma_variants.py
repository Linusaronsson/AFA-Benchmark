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
)

METHOD_TO_VARIANT_POLICY: dict[str, tuple[str, str]] = {
    "stop_baseline_marf": ("MARF", "No acquisition"),
    "stop_baseline_madt": ("MADT", "No acquisition"),
    "stop_baseline_magbt": ("MAGBT", "No acquisition"),
    "stop_baseline_malasso": ("MALasso", "No acquisition"),
    "aaco_marf": ("MARF", "AACO"),
    "aaco_madt": ("MADT", "AACO"),
    "aaco_magbt": ("MAGBT", "AACO"),
    "aaco_malasso": ("MALasso", "AACO"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot AACO improvement over MA-model no-acquisition baselines."
        )
    )
    parser.add_argument("input_path", type=Path, help="Merged parquet input.")
    parser.add_argument("output_dir", type=Path, help="Output directory.")
    return parser.parse_args()


def _compute_metrics_per_run(data: pl.DataFrame) -> pl.DataFrame:
    group_cols = [
        "afa_method",
        "dataset",
        "train_seed",
        "eval_seed",
        "eval_hard_budget",
    ]

    return data.group_by(*group_cols).map_groups(
        lambda group_df: pl.DataFrame(
            {
                "afa_method": [group_df["afa_method"].first()],
                "dataset": [group_df["dataset"].first()],
                "train_seed": [group_df["train_seed"].first()],
                "eval_seed": [group_df["eval_seed"].first()],
                "eval_hard_budget": [group_df["eval_hard_budget"].first()],
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
                "train_seed": pl.UInt64,
                "eval_seed": pl.UInt64,
                "eval_hard_budget": pl.Float64,
                "accuracy": pl.Float64,
                "f_score": pl.Float64,
            },
        )
    )


def _add_variant_policy(data: pl.DataFrame) -> pl.DataFrame:
    return data.with_columns(
        variant=pl.col("afa_method").map_elements(
            lambda method: METHOD_TO_VARIANT_POLICY[method][0],
            return_dtype=pl.String,
        ),
        policy=pl.col("afa_method").map_elements(
            lambda method: METHOD_TO_VARIANT_POLICY[method][1],
            return_dtype=pl.String,
        ),
    )


def _aggregate(metrics_df: pl.DataFrame) -> pl.DataFrame:
    with_metric = metrics_df.with_columns(
        metric=pl.when(pl.col("dataset").is_in(DATASETS_WITH_F_SCORE))
        .then(pl.col("f_score"))
        .otherwise(pl.col("accuracy"))
    )

    return (
        with_metric.group_by(
            "dataset", "variant", "policy", "eval_hard_budget"
        )
        .agg(
            mean_metric=pl.col("metric").mean(),
            std_metric=pl.col("metric").std(),
            n_runs=pl.len(),
        )
        .with_columns(
            std_metric=pl.col("std_metric").fill_null(0.0),
            low_metric=pl.col("mean_metric")
            - pl.col("std_metric").fill_null(0.0),
            high_metric=pl.col("mean_metric")
            + pl.col("std_metric").fill_null(0.0),
            variant_policy=pl.concat_str(
                [pl.col("variant"), pl.col("policy")], separator=" / "
            ),
        )
    )


def _flatten_baseline(summary_df: pl.DataFrame) -> pl.DataFrame:
    """Force no-acquisition baseline to be horizontal per dataset+variant."""
    baseline_mean = (
        pl.col("mean_metric")
        .filter(pl.col("policy") == "No acquisition")
        .first()
        .over(["dataset", "variant"])
    )
    baseline_std = (
        pl.col("std_metric")
        .filter(pl.col("policy") == "No acquisition")
        .first()
        .over(["dataset", "variant"])
    )
    baseline_low = (
        pl.col("low_metric")
        .filter(pl.col("policy") == "No acquisition")
        .first()
        .over(["dataset", "variant"])
    )
    baseline_high = (
        pl.col("high_metric")
        .filter(pl.col("policy") == "No acquisition")
        .first()
        .over(["dataset", "variant"])
    )

    return summary_df.with_columns(
        mean_metric=pl.when(pl.col("policy") == "No acquisition")
        .then(baseline_mean)
        .otherwise(pl.col("mean_metric")),
        std_metric=pl.when(pl.col("policy") == "No acquisition")
        .then(baseline_std)
        .otherwise(pl.col("std_metric")),
        low_metric=pl.when(pl.col("policy") == "No acquisition")
        .then(baseline_low)
        .otherwise(pl.col("low_metric")),
        high_metric=pl.when(pl.col("policy") == "No acquisition")
        .then(baseline_high)
        .otherwise(pl.col("high_metric")),
    )


def _add_budget_zero_anchor(summary_df: pl.DataFrame) -> pl.DataFrame:
    """Add budget=0 anchor rows so AACO curves grow from the baseline."""
    baseline = (
        summary_df.filter(pl.col("policy") == "No acquisition")
        .group_by("dataset", "variant")
        .agg(
            mean_metric=pl.col("mean_metric").first(),
            std_metric=pl.col("std_metric").first(),
            n_runs=pl.col("n_runs").first(),
            low_metric=pl.col("low_metric").first(),
            high_metric=pl.col("high_metric").first(),
        )
    )

    zero_rows = pl.concat(
        [
            baseline.with_columns(
                policy=pl.lit("No acquisition"),
                eval_hard_budget=pl.lit(0.0),
                variant_policy=pl.concat_str(
                    [pl.col("variant"), pl.lit("No acquisition")],
                    separator=" / ",
                ),
            ),
            baseline.with_columns(
                policy=pl.lit("AACO"),
                eval_hard_budget=pl.lit(0.0),
                variant_policy=pl.concat_str(
                    [pl.col("variant"), pl.lit("AACO")],
                    separator=" / ",
                ),
            ),
        ],
        how="vertical",
    ).select(summary_df.columns)

    # Keep real data if budget=0 already exists; otherwise add anchor rows.
    key_cols = ["dataset", "variant", "policy", "eval_hard_budget"]
    existing_keys = summary_df.select(key_cols).with_columns(
        exists=pl.lit(True)
    )
    new_rows = (
        zero_rows.join(existing_keys, on=key_cols, how="left")
        .filter(pl.col("exists").is_null())
        .drop("exists")
    )
    return pl.concat([summary_df, new_rows], how="vertical")


def _compute_lift(summary_df: pl.DataFrame) -> pl.DataFrame:
    pivoted = (
        summary_df.group_by("dataset", "variant", "eval_hard_budget")
        .agg(
            baseline=pl.col("mean_metric")
            .filter(pl.col("policy") == "No acquisition")
            .first(),
            aaco=pl.col("mean_metric")
            .filter(pl.col("policy") == "AACO")
            .first(),
        )
        .with_columns(lift=pl.col("aaco") - pl.col("baseline"))
    )
    return pivoted.sort(["dataset", "variant", "eval_hard_budget"])


def _make_plot(summary_df: pl.DataFrame) -> p9.ggplot:
    plot_df = summary_df.with_columns(
        dataset=pl.col("dataset").replace(DATASET_NAME_MAPPING)
    )

    n_datasets = max(plot_df["dataset"].n_unique(), 1)
    n_rows = (n_datasets + 1) // 2
    fig_height = 2.9 * n_rows

    return (
        p9.ggplot(
            plot_df,
            p9.aes(
                x="eval_hard_budget",
                y="mean_metric",
                color="variant",
                linetype="policy",
                group="variant_policy",
            ),
        )
        + p9.geom_line()
        + p9.geom_point(size=1.8)
        + p9.geom_errorbar(
            p9.aes(ymin="low_metric", ymax="high_metric"),
            width=0.08,
        )
        + p9.facet_wrap("dataset", scales="free_y", ncol=2)
        + p9.scale_color_brewer(type="qual", palette=COLOR_PALETTE_NAME)
        + p9.scale_linetype_manual(
            values={
                "No acquisition": "solid",
                "AACO": "dotted",
            }
        )
        + p9.labs(
            x="Hard budget",
            y="Metric",
            color="MA variant",
            linetype="Policy",
        )
        + p9.theme(figure_size=(11.5, fig_height))
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pl.read_parquet(args.input_path)
    required_columns = {
        "action_performed",
        "predicted_class",
        "true_class",
        "dataset",
        "afa_method",
        "train_seed",
        "eval_seed",
        "eval_hard_budget",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        msg = f"Input parquet missing columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    known_methods = set(METHOD_TO_VARIANT_POLICY)
    data = data.filter(pl.col("afa_method").is_in(known_methods))
    data = data.filter(
        (pl.col("action_performed") == 0)
        & pl.col("predicted_class").is_not_null()
        & pl.col("eval_hard_budget").is_not_null()
    )
    # We anchor budget=0 by construction to ensure MA baseline and AACO
    # coincide exactly at the starting point.
    data = data.filter(pl.col("eval_hard_budget") > 0)

    if data.is_empty():
        msg = "No rows left after filtering for MA-variant AACO plotting."
        raise ValueError(msg)

    metrics = _compute_metrics_per_run(data)
    metrics = _add_variant_policy(metrics)
    summary = _aggregate(metrics)
    summary = _flatten_baseline(summary)
    summary = _add_budget_zero_anchor(summary)
    summary = summary.sort(
        ["dataset", "variant", "policy", "eval_hard_budget"]
    )
    lift = _compute_lift(summary)

    plot = _make_plot(summary)
    plot.save(args.output_dir / "aaco_over_ma_variants.pdf")
    summary.write_csv(args.output_dir / "aaco_over_ma_variants_summary.csv")
    lift.write_csv(args.output_dir / "aaco_over_ma_variants_lift.csv")


if __name__ == "__main__":
    main()
