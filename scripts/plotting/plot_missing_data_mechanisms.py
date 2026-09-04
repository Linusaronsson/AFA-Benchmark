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

from afabench.plotting.methods import (
    INDUCED_MECHANISMS,
    MECHANISM_LABELS,
    MECHANISM_MARKERS,
    METHOD_COLORS,
    METHOD_LABELS,
    PRIMARY_METHODS,
)

# These figures are laid out one panel or one offset per primary method. The
# two reweighting controls train on the restricted view only, so they have no
# generator-quality or stepwise cell to show.
COLORS = {method: METHOD_COLORS[method] for method in PRIMARY_METHODS}
STRATEGY_LABELS = {
    "complete": "Complete",
    "restricted": "Restricted-action training",
    "pvae_label_conditioned": "Generative restoration",
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
    columns = 3 if len(COLORS) > 4 else 2
    rows = -(-len(COLORS) // columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.5 * columns, 2.1 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for axis in axes.flat[len(COLORS) :]:
        axis.set_visible(False)
    for axis, method in zip(axes.flat, COLORS, strict=False):
        method_frame = selected.loc[selected["method"] == method]
        # Marker separates the mechanism. Identification is what decides how far
        # the oracle can pull ahead, so pooling mechanisms hid the one effect
        # this figure exists to show (prop:mnar).
        for mechanism in INDUCED_MECHANISMS:
            cells = method_frame.loc[method_frame["mechanism"] == mechanism]
            if cells.empty:
                continue
            axis.scatter(
                cells["rmse_improvement"],
                cells["score_improvement"],
                color=COLORS[method],
                marker=MECHANISM_MARKERS[mechanism],
                alpha=0.5 if mechanism != "mnar_self" else 0.9,
                s=22,
                label=MECHANISM_LABELS[mechanism],
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_title(METHOD_LABELS[method], fontsize=9)
        axis.grid(alpha=0.2)
    axes.flat[0].legend(frameon=False, fontsize=6.5, loc="upper right")
    # One label per shared axis. Labelling each panel drew the top row's x-label
    # across the panel beneath it.
    fig.supxlabel("Oracle RMSE improvement")
    fig.supylabel("Oracle score improvement")
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
                "stepwise_minus_episode_start",
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
    axis.set_ylabel("Stepwise minus generative-restoration score")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    _save(fig, output, "stepwise_vs_episode_start")


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
