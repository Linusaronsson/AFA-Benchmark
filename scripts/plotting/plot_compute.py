"""Render what generative restoration costs against what it buys."""

# Both arms train for the same fixed number of batches, so this is the price of
# restoration rather than a sample-efficiency curve. The price is one generator
# pretraining, amortised over the strategy and method cells that share it, plus
# a cheap pass to restore the training view. Reads the job-level costs written
# by `scripts/analysis/collect_compute.py`.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

METHOD_COLORS = {
    "aaco": "#2a78d6",
    "ol_without_mask": "#eb6834",
    "dime": "#1baf7a",
}
METHOD_LABELS = {"aaco": "AACO", "ol_without_mask": "OL", "dime": "DIME"}
DATASET_LABELS = {"cube_nm": "CUBE-NM", "cube": "CUBE"}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#ffffff"

DIRECT = "restricted"
GENERATIVE = "pvae_label_conditioned"
RESTORED_STRATEGIES = (
    "pvae_label_conditioned",
    "pvae_label_free",
    "pvae_stepwise",
)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Typed column access, since a bare lookup widens to include ndarray."""
    return cast("pd.Series", frame[name])


def _rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Typed boolean selection, for the same reason."""
    return cast("pd.DataFrame", frame[mask])


def attribute(compute: Path) -> pd.DataFrame:
    """Total CPU seconds to produce one trained method, per cell and arm."""
    jobs = pd.read_csv(compute)
    jobs["p"] = pd.to_numeric(jobs["p"], errors="coerce")
    jobs["instance"] = pd.to_numeric(jobs["instance"], errors="coerce")

    cell = ["dataset", "mechanism", "p", "strategy", "instance"]
    training = _rows(
        jobs, _column(jobs, "rule").str.startswith("train_missing_data_method")
    )
    train_cost = training.groupby([*cell, "method"])["cpu_seconds"].sum()
    method_pretrain = (
        _rows(jobs, _column(jobs, "rule") == "pretrain_missing_data_method")
        .groupby([*cell, "method"])["cpu_seconds"]
        .sum()
    )
    view_cost = (
        _rows(
            jobs, _column(jobs, "rule") == "materialize_missing_training_view"
        )
        .groupby(cell)["cpu_seconds"]
        .sum()
    )
    restore_cost = (
        _rows(jobs, _column(jobs, "rule") == "restore_missing_training_view")
        .groupby(cell)["cpu_seconds"]
        .sum()
    )

    generator = ["dataset", "mechanism", "p", "instance"]
    generator_cost = (
        _rows(
            jobs,
            _column(jobs, "rule") == "pretrain_incomplete_restoration_pvae",
        )
        .groupby(generator)["cpu_seconds"]
        .sum()
    )
    # One generator serves every restored strategy and every method, so charge
    # each trained method only its share.
    consumers = (
        _rows(
            training, _column(training, "strategy").isin(RESTORED_STRATEGIES)
        )
        .groupby(generator)
        .size()
    )

    rows = []
    for (
        dataset,
        mechanism,
        rate,
        strategy,
        instance,
        method,
    ), training_seconds in cast("Any", train_cost).items():
        if strategy not in (DIRECT, GENERATIVE):
            continue
        key = (dataset, mechanism, rate, strategy, instance)
        total = (
            training_seconds
            + method_pretrain.get((*key, method), 0.0)
            + view_cost.get(key, 0.0)
        )
        if strategy == GENERATIVE:
            share = max(
                int(
                    consumers.get((dataset, mechanism, rate, instance), 1) or 1
                ),
                1,
            )
            total += restore_cost.get(key, 0.0)
            generator_seconds = float(
                generator_cost.get((dataset, mechanism, rate, instance), 0.0)
                or 0.0
            )
            total += generator_seconds / share
        rows.append(
            {
                "dataset": dataset,
                "mechanism": mechanism,
                "p": rate,
                "instance": instance,
                "method": method,
                "arm": "direct" if strategy == DIRECT else "generative",
                "cpu_seconds": total,
            }
        )
    return pd.DataFrame(rows)


def merge_scores(
    costs: pd.DataFrame, summary_root: Path, namespace: str
) -> pd.DataFrame:
    metrics = pd.read_csv(summary_root / namespace / "instance_metrics.csv")
    metrics = _rows(
        metrics,
        _column(metrics, "eval_hard_budget")
        == _column(metrics, "train_hard_budget"),
    )
    arm = _column(metrics, "strategy").map(
        {DIRECT: "direct", GENERATIVE: "generative"}
    )
    metrics["arm"] = arm
    metrics = _rows(metrics, _column(metrics, "arm").notna())
    keys = ["dataset", "mechanism", "p", "instance", "method", "arm"]
    scores = metrics.groupby(keys)["accuracy"].mean().reset_index()
    return costs.merge(scores, on=keys, how="inner")


def plot(frame: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": INK_MUTED,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    datasets = [d for d in ("cube_nm", "cube") if d in set(frame["dataset"])]
    figure, axes = plt.subplots(1, len(datasets), figsize=(7.0, 2.7))
    axes = [axes] if len(datasets) == 1 else list(axes)

    for axis, dataset in zip(axes, datasets, strict=True):
        per_dataset = _rows(frame, _column(frame, "dataset") == dataset)
        for method, color in METHOD_COLORS.items():
            per_method = _rows(
                per_dataset, _column(per_dataset, "method") == method
            )
            if per_method.empty:
                continue
            summary = per_method.groupby("arm").agg(
                cpu=("cpu_seconds", "median"), score=("accuracy", "mean")
            )
            if not {"direct", "generative"}.issubset(summary.index):
                continue
            direct, generative = (
                summary.loc["direct"],
                summary.loc["generative"],
            )
            axis.annotate(
                "",
                xy=(generative["cpu"], generative["score"]),
                xytext=(direct["cpu"], direct["score"]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.4,
                    "shrinkA": 3,
                    "shrinkB": 3,
                },
            )
            axis.scatter(
                [direct["cpu"]],
                [direct["score"]],
                s=20,
                facecolor=SURFACE,
                edgecolor=color,
                linewidth=1.2,
                zorder=3,
            )
            axis.scatter(
                [generative["cpu"]],
                [generative["score"]],
                s=24,
                color=color,
                zorder=4,
            )

        axis.set_xscale("log")
        axis.set_title(
            DATASET_LABELS.get(dataset, dataset),
            fontsize=8.5,
            color=INK,
            pad=4,
        )
        axis.set_xlabel(
            "CPU seconds per trained method", color=INK_MUTED, fontsize=8
        )
        axis.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("Accuracy", color=INK_MUTED, fontsize=8)

    handles = [
        Line2D([], [], color=color, linewidth=1.4, label=METHOD_LABELS[method])
        for method, color in METHOD_COLORS.items()
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            markerfacecolor=SURFACE,
            markeredgecolor=INK_MUTED,
            markeredgewidth=1.2,
            label="Direct learning (arrow head is generative)",
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7.5,
        labelcolor=INK_MUTED,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.subplots_adjust(
        left=0.09, right=0.99, top=0.90, bottom=0.31, wspace=0.18
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute",
        type=Path,
        default=Path(
            "extra/output/missing_data/analysis/compute_core_group_missingness_v1.csv"
        ),
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("extra/output/missing_data/summary/val"),
    )
    parser.add_argument("--namespace", default="core_group_missingness_v1")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extra/output/missing_data/analysis_figures/compute.pdf"),
    )
    arguments = parser.parse_args()

    costs = attribute(arguments.compute)
    frame = merge_scores(costs, arguments.summary_root, arguments.namespace)
    if frame.empty:
        message = "no cells joined"
        raise SystemExit(message)
    plot(frame, arguments.output)

    table = frame.groupby(["dataset", "method", "arm"]).agg(
        cpu=("cpu_seconds", "median"),
        score=("accuracy", "mean"),
        n=("accuracy", "size"),
    )
    print(table.round(3).to_string())
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
