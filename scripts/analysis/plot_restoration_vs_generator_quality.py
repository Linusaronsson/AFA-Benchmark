# pyright: reportCallIssue=false, reportAttributeAccessIssue=false
"""
Correlate policy degradation with PVAE generator error (H3).

Joins the per-condition accuracy summary produced by
`missing_train_analysis.py` with the imputation-quality parquets produced by
`pvae_imputation_quality.py`, on (dataset, train_initializer). Emits a tidy
CSV and a scatter plot of accuracy (retained vs the full-data baseline where
available) against imputation RMSE, one point per
(method, dataset, initializer, generator tag).

Usage:
    python scripts/analysis/plot_restoration_vs_generator_quality.py \
        --accuracy-summary extra/output/analysis/missing_train/accuracy_summary.csv \
        --quality-glob 'extra/output/analysis/pvae_quality/*.parquet' \
        --output-dir extra/output/analysis/restoration_vs_quality
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accuracy-summary",
        type=Path,
        required=True,
        help="CSV produced by missing_train_analysis.py with at least "
        "(afa_method, dataset, train_initializer, accuracy) columns.",
    )
    parser.add_argument(
        "--quality-glob",
        required=True,
        help="Glob of parquets produced by pvae_imputation_quality.py.",
    )
    parser.add_argument(
        "--label-conditioning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Which imputation mode's RMSE to use (default: "
        "label-conditioned, matching episode-start restoration).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_quality(
    quality_glob: str, *, label_conditioning: bool
) -> pd.DataFrame:
    paths = sorted(Path().glob(quality_glob))
    if not paths:
        msg = f"No quality parquets match {quality_glob!r}"
        raise FileNotFoundError(msg)
    quality_df = pd.concat(
        [pd.read_parquet(path) for path in paths], ignore_index=True
    )
    # Overall rows only (per-feature rows have a feature_index).
    quality_df = quality_df[quality_df["feature_index"].isna()]
    quality_df = quality_df[
        quality_df["label_conditioning"] == label_conditioning
    ]
    return quality_df.rename(columns={"initializer": "train_initializer"})[
        [
            "dataset",
            "train_initializer",
            "seed",
            "pvae_tag",
            "missing_fraction",
            "rmse",
        ]
    ]


def main() -> None:
    args = parse_args()
    accuracy_df = pd.read_csv(args.accuracy_summary)
    quality_df = load_quality(
        args.quality_glob, label_conditioning=args.label_conditioning
    )

    # Average generator error over mask seeds per condition.
    quality_agg = quality_df.groupby(
        ["dataset", "train_initializer", "pvae_tag"], as_index=False
    ).agg(rmse=("rmse", "mean"), missing_fraction=("missing_fraction", "mean"))

    joined = accuracy_df.merge(
        quality_agg, on=["dataset", "train_initializer"], how="inner"
    )
    if joined.empty:
        msg = (
            "Join produced no rows - check that dataset and "
            "train_initializer values match between the two inputs."
        )
        raise ValueError(msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined_path = args.output_dir / "accuracy_vs_generator_rmse.csv"
    joined.to_csv(joined_path, index=False)
    print(f"Wrote {len(joined)} joined rows to {joined_path}")

    try:
        from plotnine import (
            aes,
            facet_wrap,
            geom_point,
            geom_smooth,
            ggplot,
            labs,
            theme_bw,
        )
    except ImportError:
        print("plotnine not available; skipping plot.")
        return

    accuracy_col = (
        "accuracy" if "accuracy" in joined.columns else joined.columns[-3]
    )
    plot = (
        ggplot(
            joined,
            aes(
                x="rmse",
                y=accuracy_col,
                color="afa_method",
                shape="pvae_tag",
            ),
        )
        + geom_point()
        + geom_smooth(method="lm", se=False, linetype="dashed")
        + facet_wrap("~dataset", scales="free")
        + labs(
            x="PVAE imputation RMSE (mechanism-missing entries)",
            y=accuracy_col,
            title="Policy performance vs generator error",
        )
        + theme_bw()
    )
    plot_path = args.output_dir / "accuracy_vs_generator_rmse.png"
    plot.save(plot_path, width=10, height=6, dpi=200, verbose=False)
    print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
