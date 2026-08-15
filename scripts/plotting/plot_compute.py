"""Plot paired restoration gain against generative/direct wall-time ratio."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.lines import Line2D

from afabench.plotting.methods import (
    DATASET_LABELS_SHORT,
    LEGEND_STRIP_IN,
    METHOD_COLORS,
    METHOD_LABELS,
    PRIMARY_METHODS,
    TEXT_WIDTH_IN,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

DATASET_MARKERS = {
    "cube": "o",
    "cube_nm": "s",
    "cube_nonuniform_costs": "D",
    "heart_disease": "P",
    "actg": "^",
    "diabetes": "v",
    "nhanes_mortality": "X",
}
ACCURACY_DATASETS = {"cube", "cube_nm", "cube_nonuniform_costs"}
DIRECT = "restricted"
GENERATIVE = "pvae_label_conditioned"
RESTORED = {"pvae_label_conditioned", "pvae_label_free", "pvae_stepwise"}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#ffffff"


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast("pd.Series", frame[name])


def _rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return cast("pd.DataFrame", frame[mask])


def _sum_cost(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return cast(
        "pd.DataFrame",
        frame.groupby(keys, dropna=False).agg(
            wall_seconds=("wall_seconds", "sum"),
            cpu_seconds=("cpu_seconds", "sum"),
            peak_rss_mb=("peak_rss_mb", "max"),
        ),
    )


def attribute(compute: pd.DataFrame) -> pd.DataFrame:
    """Cost one direct or generative trained-method result within each cell."""
    cell = [
        "namespace",
        "dataset",
        "mechanism",
        "p",
        "strategy",
        "instance",
    ]
    env = [
        "hardware_signature",
        "git_commit",
        "device",
        "cores",
        "mem_mb",
        "gpu_workers",
        "mps",
        "architecture",
        "torch",
        "torch_cuda",
    ]
    training = _rows(compute, _column(compute, "rule") == "train_method")
    training = _rows(
        training, _column(training, "method").isin(PRIMARY_METHODS)
    )
    pretraining = _rows(
        compute, _column(compute, "rule") == "pretrain_method"
    ).copy()
    pretraining["method"] = pretraining["pretrain_key"]
    evaluations = _rows(compute, _column(compute, "rule") == "eval_method")
    restoration = _rows(compute, _column(compute, "rule") == "restore_view")
    generators = _rows(
        compute,
        _column(compute, "rule") == "pretrain_restoration_pvae_incomplete",
    )

    method_keys = [*cell, "method", *env]
    train_cost = _sum_cost(training, method_keys)
    eval_cost = _sum_cost(evaluations, method_keys)
    pretrain_cost = _sum_cost(pretraining, method_keys)
    generator_keys = [
        "namespace",
        "dataset",
        "mechanism",
        "p",
        "instance",
        *env,
    ]
    generator_cost = _sum_cost(generators, generator_keys)
    restore_cost = _sum_cost(restoration, cell)
    consumers = (
        _rows(training, _column(training, "strategy").isin(RESTORED))
        .groupby(generator_keys, dropna=False)
        .size()
    )
    strategy_consumers = training.groupby(cell, dropna=False).size()

    rows: list[dict[str, Any]] = []
    for key, train in cast("Any", train_cost).iterrows():
        record = dict(zip(method_keys, key, strict=True))
        strategy = record["strategy"]
        if strategy not in {DIRECT, GENERATIVE}:
            continue
        components = [train]
        if key in pretrain_cost.index:
            components.append(pretrain_cost.loc[key])
        if key in eval_cost.index:
            components.append(eval_cost.loc[key])
        if strategy == GENERATIVE:
            generator_key = tuple(record[name] for name in generator_keys)
            if generator_key in generator_cost.index:
                share = max(
                    int(cast("Any", consumers.get(generator_key, 1)) or 1),
                    1,
                )
                components.append(generator_cost.loc[generator_key] / share)
            strategy_key = tuple(record[name] for name in cell)
            if strategy_key in restore_cost.index:
                share = max(
                    int(
                        cast("Any", strategy_consumers.get(strategy_key, 1))
                        or 1
                    ),
                    1,
                )
                components.append(restore_cost.loc[strategy_key] / share)
        rows.append(
            {
                **record,
                "arm": "direct" if strategy == DIRECT else "generative",
                "wall_seconds": sum(
                    float(part["wall_seconds"]) for part in components
                ),
                "cpu_seconds": sum(
                    float(part["cpu_seconds"]) for part in components
                ),
                "peak_rss_mb": max(
                    float(part["peak_rss_mb"]) for part in components
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_costs(costs: pd.DataFrame, summary_root: Path) -> pd.DataFrame:
    scores = []
    for namespace in sorted(set(_column(costs, "namespace"))):
        path = summary_root / str(namespace) / "instance_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["namespace"] = namespace
        frame["arm"] = _column(frame, "strategy").map(
            {DIRECT: "direct", GENERATIVE: "generative"}
        )
        for dataset in set(_column(frame, "dataset")):
            metric = "accuracy" if dataset in ACCURACY_DATASETS else "f_score"
            subset = _rows(frame, _column(frame, "dataset") == dataset).copy()
            subset["score"] = subset[metric]
            scores.append(subset)
    if not scores:
        return pd.DataFrame()
    score_frame = pd.concat(scores, ignore_index=True)
    keys = [
        "namespace",
        "dataset",
        "mechanism",
        "p",
        "instance",
        "method",
        "arm",
    ]
    joined = costs.merge(score_frame[[*keys, "score"]], on=keys, how="inner")
    index = [
        "namespace",
        "dataset",
        "mechanism",
        "p",
        "instance",
        "method",
        "hardware_signature",
        "git_commit",
        "device",
        "cores",
        "mem_mb",
        "gpu_workers",
        "mps",
        "architecture",
        "torch",
        "torch_cuda",
    ]
    wide = joined.pivot_table(
        index=index,
        columns="arm",
        values=["wall_seconds", "cpu_seconds", "peak_rss_mb", "score"],
    )
    wide.columns = [f"{measure}_{arm}" for measure, arm in wide.columns]
    wide = wide.reset_index()
    required = [
        "wall_seconds_direct",
        "wall_seconds_generative",
        "score_direct",
        "score_generative",
    ]
    wide = wide.dropna(subset=required)
    wide["wall_time_ratio"] = (
        wide["wall_seconds_generative"] / wide["wall_seconds_direct"]
    )
    wide["restoration_gain"] = wide["score_generative"] - wide["score_direct"]
    return wide


def _thin_x_ticks(axis: Axes, keep: int = 3) -> None:
    """
    Drop every other label until at most ``keep`` remain.

    A 1-2-5 log locator gives a sensible number of ticks over half a decade and
    far too many over one and a half, which is the span between the cheapest and
    dearest method on Diabetes.
    """
    axis.figure.canvas.draw()
    lo, hi = axis.get_xlim()
    visible = [t for t in axis.get_xticks() if lo <= t <= hi]
    while len(visible) > keep:
        visible = visible[::2]
    axis.set_xticks(visible)


PANEL_MECHANISM, PANEL_RATE = "mcar", 0.5


def panel_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """
    One direct and one generative point per dataset and method.

    Fixed to the same reference cell as the dumbbell panel of the main figure,
    so the two figures describe one cell rather than two different ones, and
    averaged over the five dataset instances.
    """
    cell = _rows(
        frame,
        (_column(frame, "mechanism") == PANEL_MECHANISM)
        & (_column(frame, "p") == PANEL_RATE),
    )
    return cast(
        "pd.DataFrame",
        cell.groupby(["dataset", "method"], as_index=False)[
            [
                "wall_seconds_direct",
                "wall_seconds_generative",
                "score_direct",
                "score_generative",
            ]
        ].mean(),
    )


def plot(frame: pd.DataFrame, output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.6,
            "axes.edgecolor": GRID,
            "text.color": INK,
            "axes.labelcolor": INK_MUTED,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )
    cells = panel_cells(frame)
    datasets = sorted(set(_column(cells, "dataset")))
    # Four columns past six datasets, so eight fills a 2x4 exactly rather than
    # leaving a hole in a 3x3 and a third of a page of white.
    columns = 4 if len(datasets) > 6 else min(3, len(datasets))
    rows = -(-len(datasets) // columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(TEXT_WIDTH_IN, 1.35 + 1.55 * rows),
        squeeze=False,
    )
    for index, dataset in enumerate(datasets):
        axis = axes[index // columns][index % columns]
        per_dataset = _rows(cells, _column(cells, "dataset") == dataset)
        for method in PRIMARY_METHODS:
            record = _rows(
                per_dataset, _column(per_dataset, "method") == method
            )
            if record.empty:
                continue
            row = cast("Any", record.iloc[0])
            color = METHOD_COLORS[method]
            # An arrow from what direct learning bought to what restoration
            # bought, in the plane the question is actually asked in: does the
            # generative arm buy score, and at what multiple of the cost.
            axis.annotate(
                "",
                xy=(row["wall_seconds_generative"], row["score_generative"]),
                xytext=(row["wall_seconds_direct"], row["score_direct"]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": color,
                    "linewidth": 1.3,
                    "shrinkA": 2.0,
                    "shrinkB": 2.0,
                },
            )
            axis.scatter(
                [row["wall_seconds_direct"]],
                [row["score_direct"]],
                s=16,
                facecolor=SURFACE,
                edgecolor=color,
                linewidth=1.1,
                zorder=3,
            )
            # An annotation arrow contributes nothing to the data limits, so
            # autoscaling from the direct endpoints alone cropped every arrow
            # whose head landed further right. Register the head too, invisibly.
            axis.scatter(
                [row["wall_seconds_generative"]],
                [row["score_generative"]],
                s=0,
                alpha=0.0,
            )
        axis.set_xscale("log")
        # Wall time spans well under a decade per dataset. The default locator
        # crowds the axis with 2x10^2, 3x10^2 and so on, and restricting it to
        # decades leaves panels with no labelled tick at all, so tick the 1-2-5
        # subdivisions.
        axis.xaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12)
        )
        axis.xaxis.set_minor_locator(mticker.NullLocator())
        axis.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda value, _: f"{value:,.0f}")
        )
        axis.tick_params(labelsize=7)
        # A 1.2in panel at four columns cannot carry three log labels like
        # "10,000" without them touching, so thin harder as the grid widens.
        _thin_x_ticks(axis, keep=2 if columns > 3 else 3)
        axis.set_title(DATASET_LABELS_SHORT.get(dataset, dataset), fontsize=8)
        axis.grid(True, color=GRID, linewidth=0.4, alpha=0.55)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    for index in range(len(datasets), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    height = 1.35 + 1.55 * rows
    figure.supxlabel(
        "Wall-clock time per trained method (s)",
        fontsize=8,
        y=LEGEND_STRIP_IN * 0.65 / height,
    )
    figure.supylabel("Accuracy or macro-F1", fontsize=8, x=0.015)
    handles = [
        Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            linewidth=1.3,
            label=METHOD_LABELS[method],
        )
        for method in PRIMARY_METHODS
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
            markeredgewidth=1.0,
            label="Direct learning",
        )
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=6.5,
        labelcolor=INK_MUTED,
        columnspacing=1.2,
        handlelength=1.6,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.92,
        bottom=LEGEND_STRIP_IN / height,
        hspace=0.45,
        wspace=0.32,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".svg"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute", type=Path, required=True)
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("extra/output/missing_data/summary/val"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extra/output/missing_data/analysis_figures/compute.pdf"),
    )
    parser.add_argument("--table", type=Path)
    arguments = parser.parse_args()

    costs = attribute(pd.read_csv(arguments.compute))
    frame = paired_costs(costs, arguments.summary_root)
    if frame.empty:
        message = "no complete paired compute cells"
        raise SystemExit(message)
    plot(frame, arguments.output)
    table = arguments.table or arguments.output.with_suffix(".paired.csv")
    if table.resolve() == arguments.compute.resolve():
        message = "paired table must not overwrite the raw compute input"
        raise ValueError(message)
    frame.to_csv(table, index=False)
    print(f"paired cells: {len(frame)}")
    print(f"wrote {arguments.output} and {table}")


if __name__ == "__main__":
    main()
