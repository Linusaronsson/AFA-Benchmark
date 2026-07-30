"""Render compact confirmatory mechanism figures from paired CSV tables."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.figure import Figure

COLORS = {
    "aaco": "#1B9E77",
    "dime": "#7570B3",
    "ol_with_mask": "#E69F00",
    "ol_full_state": "#D55E00",
}
METHOD_LABELS = {
    "aaco": "AACO",
    "dime": "DIME",
    "ol_with_mask": "OL (mask)",
    "ol_full_state": "OL (full state)",
}
STRATEGY_LABELS = {
    "complete": "Complete",
    "restricted": "Restricted",
    "pvae_label_conditioned": "One-shot",
    "pvae_stepwise": "Stepwise",
}


def _save(fig: Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(
            output / f"{name}.{suffix}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def _read_analysis(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def plot_path_fidelity(frame: pd.DataFrame, output: Path) -> None:
    selected = frame.loc[
        frame["method"].isin(COLORS)
        & (
            (
                (frame["mechanism"] == "none")
                & (frame["strategy"] == "complete")
            )
            | (
                (frame["mechanism"] == "mcar")
                & np.isclose(frame["p"], 0.7)
                & frame["strategy"].isin(
                    [
                        "restricted",
                        "pvae_label_conditioned",
                        "pvae_stepwise",
                    ]
                )
            )
        )
    ].copy()
    strategy_order = list(STRATEGY_LABELS)
    metrics = [
        ("context_first_rate", "Context acquired first"),
        ("correct_next_rate", "Correct branch next"),
        ("correct_block_allocation", "Acquisitions in correct branch"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2), sharey=True)
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        method_offsets = np.linspace(-0.24, 0.24, len(COLORS))
        for method_offset, method in zip(method_offsets, COLORS, strict=True):
            method_frame = selected.loc[selected["method"] == method]
            for strategy_index, strategy in enumerate(strategy_order):
                values = method_frame.loc[
                    method_frame["strategy"] == strategy,
                    metric,
                ].dropna()
                if values.empty:
                    continue
                x = strategy_index + method_offset
                jitter = np.linspace(-0.035, 0.035, len(values))
                axis.scatter(
                    x + jitter,
                    values,
                    color=COLORS[method],
                    alpha=0.55,
                    s=18,
                )
                axis.scatter(
                    x,
                    values.mean(),
                    color=COLORS[method],
                    edgecolor="black",
                    linewidth=0.5,
                    s=50,
                    label=METHOD_LABELS[method]
                    if strategy_index == 0 and axis is axes[0]
                    else None,
                )
        axis.set_title(title)
        axis.set_xticks(range(len(strategy_order)))
        axis.set_xticklabels(
            [STRATEGY_LABELS[value] for value in strategy_order],
            rotation=25,
            ha="right",
        )
        axis.set_ylim(-0.03, 1.03)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Episode-level rate")
    axes[0].legend(frameon=False)
    _save(fig, output, "cube_nm_path_fidelity")


def plot_generator_quality(frame: pd.DataFrame, output: Path) -> None:
    selected = frame.loc[frame["method"].isin(COLORS)].copy()
    fig, axes = plt.subplots(
        2, 2, figsize=(7.4, 6.2), sharex=True, sharey=True
    )
    for axis, method in zip(axes.flat, COLORS, strict=True):
        method_frame = selected.loc[selected["method"] == method]
        axis.scatter(
            method_frame["rmse_improvement"],
            method_frame["score_improvement"],
            color=COLORS[method],
            alpha=0.65,
            s=24,
        )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_title(METHOD_LABELS[method])
        axis.set_xlabel("Oracle RMSE improvement")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Oracle score improvement")
    axes[1, 0].set_ylabel("Oracle score improvement")
    _save(fig, output, "generator_quality_vs_score")


def plot_stepwise(frame: pd.DataFrame, output: Path) -> None:
    selected = frame.loc[frame["method"].isin(COLORS)].copy()
    datasets = selected["dataset"].drop_duplicates().tolist()
    fig, axis = plt.subplots(figsize=(max(4.2, len(datasets) * 1.8), 3.2))
    method_offsets = np.linspace(-0.24, 0.24, len(COLORS))
    for method_offset, method in zip(method_offsets, COLORS, strict=True):
        for dataset_index, dataset in enumerate(datasets):
            values = selected.loc[
                (selected["method"] == method)
                & (selected["dataset"] == dataset),
                "stepwise_minus_one_shot",
            ].dropna()
            if values.empty:
                continue
            x = dataset_index + method_offset
            jitter = np.linspace(-0.035, 0.035, len(values))
            axis.scatter(
                x + jitter,
                values,
                color=COLORS[method],
                alpha=0.55,
                s=18,
            )
            axis.scatter(
                x,
                values.mean(),
                color=COLORS[method],
                edgecolor="black",
                linewidth=0.5,
                s=50,
                label=METHOD_LABELS[method] if dataset_index == 0 else None,
            )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(datasets)))
    axis.set_xticklabels(datasets)
    axis.set_ylabel("Stepwise minus one-shot score")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _save(fig, output, "stepwise_vs_one_shot")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    suffix = f"{args.namespace}_{args.split}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analyses = [
        (
            plot_path_fidelity,
            args.analysis_dir / f"path_fidelity_{suffix}",
        ),
        (
            plot_generator_quality,
            args.analysis_dir / f"generator_quality_{suffix}",
        ),
        (
            plot_stepwise,
            args.analysis_dir / f"stepwise_effects_{suffix}",
        ),
    ]
    for plot, path in analyses:
        frame = _read_analysis(path)
        if not frame.empty:
            plot(frame, args.output_dir)


if __name__ == "__main__":
    main()
