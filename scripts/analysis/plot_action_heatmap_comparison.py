# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""
Side-by-side action heatmaps: full-data vs missing-data policies.

Compares how a policy's feature-selection pattern changes when trained
under missingness, using the same heatmap style as plot_eval_actions.py.

Usage:
    python scripts/analysis/plot_action_heatmap_comparison.py \
        --results-dir extra/output/merged_results \
        --method aaco_full \
        --dataset afa_context_v2 \
        --comparison-initializers mcar_p03 mar_p03 mnar_logistic_p03 \
        --output-dir extra/output/analysis/missing_train/heatmaps
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from afabench.eval.plotting_config import (
    DATASET_NAME_MAPPING,
    METHOD_NAME_MAPPING,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from numpy.typing import NDArray


INITIALIZER_LABELS: dict[str, str] = {
    "cold": "Full data",
    "mcar_p01": "MCAR p=0.1",
    "mcar_p03": "MCAR p=0.3",
    "mcar_p05": "MCAR p=0.5",
    "mcar_p07": "MCAR p=0.7",
    "mar_p01": "MAR p=0.1",
    "mar_p03": "MAR p=0.3",
    "mar_p05": "MAR p=0.5",
    "mar_p07": "MAR p=0.7",
    "mnar_logistic_p01": "MNAR p=0.1",
    "mnar_logistic_p03": "MNAR p=0.3",
    "mnar_logistic_p05": "MNAR p=0.5",
    "mnar_logistic_p07": "MNAR p=0.7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("extra/output/merged_results"),
    )
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--method", default="aaco_full")
    parser.add_argument("--dataset", default="afa_context_v2")
    parser.add_argument("--baseline-initializer", default="cold")
    parser.add_argument(
        "--comparison-initializers",
        nargs="+",
        default=["mcar_p03", "mar_p03", "mnar_logistic_p03"],
    )
    parser.add_argument("--method-set", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extra/output/analysis/missing_train/heatmaps"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["pdf"],
    )
    return parser.parse_args()


def _find_parquet(
    results_dir: Path,
    eval_split: str,
    train_init: str,
    method_set: str | None,
) -> Path | None:
    """Locate the merged parquet for a given train initializer."""
    split_dir = results_dir / f"eval_split-{eval_split}"
    if not split_dir.exists():
        return None
    for init_dir in split_dir.iterdir():
        if f"train_initializer-{train_init}" in init_dir.name:
            perf_dir = init_dir / "eval_perf"
            if not perf_dir.exists():
                continue
            if method_set is not None:
                candidate = perf_dir / f"method_set-{method_set}+all.parquet"
                if candidate.exists():
                    return candidate
            # Fallback: pick first parquet.
            parquets = sorted(perf_dir.glob("*.parquet"))
            if parquets:
                return parquets[0]
    return None


def _normalize_heatmap(
    df: pl.DataFrame,
    max_action: int,
    max_time: int,
) -> NDArray[np.float64]:
    """Build timestep-normalized heatmap, excluding action 0."""
    heatmap = np.zeros((max_action, max_time + 1))
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


def _format_axes(
    ax: Axes,
    max_action: int,
    max_time: int,
    title: str,
) -> None:
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Action index")
    ax.set_xticks(np.arange(0, max_time + 1, max(1, max_time // 5)))
    y_ticks = np.arange(0, max_action + 1, max(1, (max_action + 1) // 10))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks + 1)


def plot_heatmap_comparison(
    baseline_df: pl.DataFrame,
    comparison_dfs: dict[str, pl.DataFrame],
    *,
    method: str,
    dataset: str,
    output_dir: Path,
    formats: Sequence[str] = ("pdf",),
) -> None:
    """Create side-by-side heatmaps for baseline vs each comparison."""
    all_dfs = [baseline_df, *comparison_dfs.values()]
    global_max_action = max(
        cast("int", df["action_performed"].max())
        for df in all_dfs
        if len(df) > 0
    )
    global_max_time = max(
        cast("int", df["n_selections_performed"].max())
        for df in all_dfs
        if len(df) > 0
    )

    n_panels = 1 + len(comparison_dfs)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(4.5 * n_panels, 4),
        squeeze=False,
    )

    # Baseline panel.
    heatmap = _normalize_heatmap(
        baseline_df, global_max_action, global_max_time
    )
    axes[0, 0].imshow(
        heatmap,
        cmap="Blues",
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    _format_axes(
        axes[0, 0],
        global_max_action,
        global_max_time,
        INITIALIZER_LABELS.get("cold", "Full data"),
    )

    # Comparison panels.
    for idx, (init_name, df) in enumerate(comparison_dfs.items(), start=1):
        heatmap = _normalize_heatmap(df, global_max_action, global_max_time)
        axes[0, idx].imshow(
            heatmap,
            cmap="Blues",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )
        title = INITIALIZER_LABELS.get(init_name, init_name.replace("_", " "))
        _format_axes(axes[0, idx], global_max_action, global_max_time, title)

    method_label = METHOD_NAME_MAPPING.get(method, method)
    dataset_label = DATASET_NAME_MAPPING.get(dataset, dataset)
    fig.suptitle(
        f"{method_label} on {dataset_label}: "
        f"Policy Behavior Under Training Missingness",
        fontsize=11,
    )
    fig.tight_layout()

    for fmt in formats:
        path = output_dir / f"heatmap_comparison_{method}_{dataset}.{fmt}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline.
    bl_path = _find_parquet(
        args.results_dir,
        args.eval_split,
        args.baseline_initializer,
        args.method_set,
    )
    if bl_path is None:
        print(
            f"Baseline parquet not found for "
            f"initializer={args.baseline_initializer}."
        )
        return
    bl_df = pl.read_parquet(bl_path)
    bl_df = bl_df.filter(
        (pl.col("afa_method") == args.method)
        & (pl.col("dataset") == args.dataset)
        & (pl.col("action_performed") != 0)
    )
    # Keep only largest hard budget.
    max_budget = bl_df["eval_hard_budget"].max()
    bl_df = bl_df.filter(pl.col("eval_hard_budget") == max_budget)
    if bl_df.is_empty():
        print(
            f"No baseline data for method={args.method}, "
            f"dataset={args.dataset}."
        )
        return

    # Load comparison initializers.
    comparison_dfs: dict[str, pl.DataFrame] = {}
    for init_name in args.comparison_initializers:
        path = _find_parquet(
            args.results_dir,
            args.eval_split,
            init_name,
            args.method_set,
        )
        if path is None:
            print(
                f"Warning: parquet not found for "
                f"initializer={init_name}, skipping."
            )
            continue
        comparison_df = pl.read_parquet(path)
        comparison_df = comparison_df.filter(
            (pl.col("afa_method") == args.method)
            & (pl.col("dataset") == args.dataset)
            & (pl.col("action_performed") != 0)
        )
        comparison_df = comparison_df.filter(
            pl.col("eval_hard_budget") == max_budget
        )
        if comparison_df.is_empty():
            print(f"Warning: no data for {init_name}, skipping.")
            continue
        comparison_dfs[init_name] = comparison_df

    if not comparison_dfs:
        print("No comparison data found. Nothing to plot.")
        return

    plot_heatmap_comparison(
        bl_df,
        comparison_dfs,
        method=args.method,
        dataset=args.dataset,
        output_dir=args.output_dir,
        formats=args.formats,
    )

    print("Done.")


if __name__ == "__main__":
    main()
