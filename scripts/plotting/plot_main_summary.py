"""Render the paper's main figure, direct learning against generative restoration."""

# Writes three figures: the per-dataset levels in raw units and in gap units,
# and the restoration gain against the missingness damage over every cell.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D

from afabench.plotting.methods import (
    DATASET_LABELS,
    GRID,
    INK,
    INK_MUTED,
    LEGEND_STRIP_IN,
    MECHANISM_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    PRIMARY_METHODS,
    SURFACE,
    TEXT_WIDTH_IN,
    WEDGE,
    apply_paper_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


ACCURACY_DATASETS = {"cube", "cube_nm", "cube_nonuniform_costs"}

# Current induced-missingness confirmatory namespaces. The factual-native arm
# has neither an induced MCAR 0.5 cell nor a counterfactual complete-data
# ceiling, so it belongs in a separate panel rather than this figure.
SOURCES = {
    "core_group_missingness_v2": ["cube_nm", "cube"],
    "induced_nonuniform_missingness_v2": [
        "cube_nonuniform_costs",
        "heart_disease",
    ],
    "induced_real_missingness_v2": [
        "actg",
        "diabetes",
        "nhanes_mortality",
        "miniboone",
    ],
}


DIRECT = "restricted"
GENERATIVE = "pvae_label_conditioned"
PANEL_MECHANISM, PANEL_RATE = "mcar", 0.5


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Typed column access, since a bare lookup widens to include ndarray."""
    return cast("pd.Series", frame[name])


def _rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    """Typed boolean selection, for the same reason."""
    return cast("pd.DataFrame", frame[mask])


def primary_metric(dataset: str) -> str:
    return "accuracy" if dataset in ACCURACY_DATASETS else "f_score"


def _largest_budget(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    subset = _rows(frame, _column(frame, "dataset") == dataset)
    budget = _column(subset, "eval_hard_budget")
    return _rows(subset, budget == budget.max())


BOOTSTRAP_DRAWS = 2000
SEED = 0


def _interval(
    values: npt.NDArray[np.float64], rng: np.random.Generator
) -> tuple[float, float]:
    """
    Percentile bootstrap over the dataset instances.

    Five instances is few, which is the regime stratified bootstrap intervals
    exist for; a normal-theory standard error would be assuming more than the
    data supports.
    """
    if len(values) < 2:
        return (float("nan"), float("nan"))
    draws = values[
        rng.integers(0, len(values), (BOOTSTRAP_DRAWS, len(values)))
    ]
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _level_row(
    per_method: pd.DataFrame,
    metric: str,
    gap_column: str,
    rate: float,
    rng: np.random.Generator,
) -> dict[str, float] | None:
    """One method's direct and restored levels at one rate, in both units."""

    def cells(strategy: str) -> pd.DataFrame:
        return _rows(
            per_method,
            (_column(per_method, "strategy") == strategy)
            & (_column(per_method, "mechanism") == PANEL_MECHANISM)
            & (_column(per_method, "p") == rate),
        )

    columns: dict[str, float] = {}
    for key, strategy in (("direct", DIRECT), ("generative", GENERATIVE)):
        selected = cells(strategy)
        # The gap column is written as score - complete, so negate it.
        gap = -_column(selected, gap_column).dropna().to_numpy()
        score = _column(selected, metric).dropna().to_numpy()
        if not len(gap) or not len(score):
            return None
        for name, values in ((key, gap), (f"{key}_abs", score)):
            low, high = _interval(values, rng)
            columns[name] = float(values.mean())
            columns[f"{name}_lo"] = low
            columns[f"{name}_hi"] = high
    return columns


def collect_levels(summary_root: Path) -> pd.DataFrame:
    """
    Where each method lands under missingness, per rate, both ways.

    The gap columns are the distance below that method's own complete-data
    ceiling, which is what lets every dataset share one scale on which zero means
    the same thing. The `_abs` columns are the same cells in the dataset's raw
    primary metric, together with the ceiling itself, for the absolute view where
    each dataset keeps its own scale.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for namespace, datasets in SOURCES.items():
        path = summary_root / namespace / "instance_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for dataset in datasets:
            per_dataset = _largest_budget(frame, dataset)
            metric = primary_metric(dataset)
            # summarize_missing_data.py writes score - complete, so negate it.
            gap_column = f"{metric}_gap_to_complete"
            for method in PRIMARY_METHODS:
                per_method = _rows(
                    per_dataset, _column(per_dataset, "method") == method
                )
                if per_method.empty:
                    continue
                ceiling = float(
                    _column(
                        _rows(
                            per_method,
                            _column(per_method, "strategy") == "complete",
                        ),
                        metric,
                    ).mean()
                )
                rates = sorted(
                    {
                        float(value)
                        for value in _column(per_method, "p")
                        if float(value) > 0.0
                    }
                )
                for rate in rates:
                    record = _level_row(
                        per_method, metric, gap_column, rate, rng
                    )
                    if record is None:
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "p": rate,
                            "metric": metric,
                            "ceiling_abs": ceiling,
                            **record,
                        }
                    )
    return pd.DataFrame(rows)


def collect_panel_b(summary_root: Path) -> pd.DataFrame:
    rows = []
    for namespace, datasets in SOURCES.items():
        path = summary_root / namespace / "instance_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for dataset in datasets:
            per_dataset = _largest_budget(frame, dataset)
            metric = primary_metric(dataset)
            for method in PRIMARY_METHODS:
                per_method = _rows(
                    per_dataset, _column(per_dataset, "method") == method
                )
                if per_method.empty:
                    continue
                complete = _rows(
                    per_method, _column(per_method, "strategy") == "complete"
                )[metric].mean()
                grouped = cast("Any", per_method.groupby(["mechanism", "p"]))
                for (mechanism, rate), cell in grouped:
                    if mechanism == "none":
                        continue
                    direct = _rows(cell, _column(cell, "strategy") == DIRECT)[
                        metric
                    ].mean()
                    generative = _rows(
                        cell, _column(cell, "strategy") == GENERATIVE
                    )[metric].mean()
                    triple = [
                        float(complete),
                        float(direct),
                        float(generative),
                    ]
                    if np.isnan(triple).any():
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "mechanism": mechanism,
                            "p": rate,
                            "damage": complete - direct,
                            "gain": generative - direct,
                        }
                    )
    return pd.DataFrame(rows)


def _draw_levels(
    axis: Axes, levels: pd.DataFrame, dataset: str, *, absolute: bool = False
) -> None:
    """One dataset's panel: a dumbbell per method at each missingness rate."""
    suffix = "_abs" if absolute else ""
    per_dataset = _rows(levels, _column(levels, "dataset") == dataset)
    rates = sorted({float(value) for value in _column(per_dataset, "p")})
    offsets = np.linspace(-0.26, 0.26, len(PRIMARY_METHODS))
    for offset, method in zip(offsets, PRIMARY_METHODS, strict=True):
        per_method = _rows(
            per_dataset, _column(per_dataset, "method") == method
        )
        color = METHOD_COLORS[method]
        for _, record in per_method.iterrows():
            x = rates.index(float(record["p"])) + offset
            for key in (f"direct{suffix}", f"generative{suffix}"):
                low, high = record[f"{key}_lo"], record[f"{key}_hi"]
                if np.isnan(low):
                    continue
                axis.plot(
                    [x, x],
                    [low, high],
                    color=color,
                    linewidth=3.0,
                    alpha=0.22,
                    solid_capstyle="butt",
                    zorder=1,
                )
            axis.plot(
                [x, x],
                [record[f"direct{suffix}"], record[f"generative{suffix}"]],
                color=color,
                linewidth=1.3,
                solid_capstyle="round",
                zorder=2,
            )
            axis.scatter(
                [x],
                [record[f"direct{suffix}"]],
                s=13,
                facecolor=SURFACE,
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
            axis.scatter(
                [x],
                [record[f"generative{suffix}"]],
                s=16,
                color=color,
                zorder=4,
            )
            if absolute:
                # In raw units the ceiling is a different number for every
                # method, so it cannot be one line across the panel. A tick at
                # the method's own offset keeps each dumbbell beside the target
                # it is trying to reach.
                axis.plot(
                    [x - 0.075, x + 0.075],
                    [record["ceiling_abs"]] * 2,
                    color=color,
                    linewidth=1.1,
                    solid_capstyle="butt",
                    zorder=5,
                )
    if not absolute:
        # Zero is the complete-data ceiling for every dataset, which is the
        # point of plotting a gap: one scale on which zero means the same thing.
        axis.axhline(0.0, color=INK_MUTED, linewidth=0.9, zorder=5)
    axis.set_xticks(range(len(rates)))
    axis.set_xticklabels([f"{rate:g}" for rate in rates], fontsize=7)
    axis.set_xlim(-0.55, len(rates) - 0.45)
    axis.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=8)
    axis.grid(True, axis="y", color=GRID, linewidth=0.4, alpha=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=7)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def _draw_law(axis_b: Axes, panel_b: pd.DataFrame) -> None:
    lo = min(panel_b["damage"].min(), panel_b["gain"].min()) - 0.03
    hi = max(panel_b["damage"].max(), panel_b["gain"].max()) + 0.03
    # The wedge and the diagonal run to the right edge rather than stopping at
    # the data, so the reference geometry does not appear to be cut off where
    # the tip labels begin.
    edge = hi + 0.34 * (hi - lo)
    # Name the two reference lines, so a point's height is read directly rather
    # than decoded from the definitions of D and R. Everything between them is
    # partial recovery; below zero restoration made the method worse.
    axis_b.fill_between(
        [0.0, hi],
        [0.0, 0.0],
        [0.0, hi],
        color=WEDGE,
        linewidth=0,
        zorder=0,
    )
    axis_b.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=0.9, zorder=2)
    axis_b.axhline(0.0, color=INK_MUTED, linewidth=0.9, zorder=2)
    axis_b.axvline(0.0, color=GRID, linewidth=0.8, zorder=1)
    axis_b.annotate(
        "full recovery, $R=D$",
        (hi, hi),
        textcoords="offset points",
        xytext=(-3, -9),
        fontsize=6.5,
        color=INK_MUTED,
        ha="right",
    )
    axis_b.annotate(
        "nothing recovered, $R=0$",
        (edge, 0.0),
        textcoords="offset points",
        xytext=(-3, -6),
        fontsize=6.5,
        color=INK_MUTED,
        ha="right",
        va="top",
    )
    # Marker as well as colour, so identity survives a greyscale print.
    for method in PRIMARY_METHODS:
        subset = _rows(panel_b, _column(panel_b, "method") == method)
        if subset.empty:
            continue
        # The fitted share as a ray, its bootstrap interval as a fan.
        share, low, high = _share(panel_b, method)
        if not np.isnan(share):
            # Each ray stops at that method's own largest damage, so its
            # length shows the lever arm the share was fitted over.
            span = float(_column(subset, "damage").max())
            axis_b.fill_between(
                [0.0, span],
                [0.0, low * span],
                [0.0, high * span],
                color=METHOD_COLORS[method],
                alpha=0.16,
                linewidth=0,
                zorder=2,
            )
            axis_b.plot(
                [0.0, span],
                [0.0, share * span],
                color=METHOD_COLORS[method],
                linewidth=1.4,
                solid_capstyle="round",
                # Cased, or the short rays vanish into the cluster at the
                # origin that they summarise.
                path_effects=[
                    patheffects.Stroke(linewidth=2.8, foreground=SURFACE),
                    patheffects.Normal(),
                ],
                zorder=4,
            )
            # Direct labels at the tip, so four series need no legend. A short
            # ray's label would land mid-plot on its neighbour, so those are
            # lifted above their own ray instead of trailing it.
            widest = max(
                float(
                    _column(
                        _rows(panel_b, _column(panel_b, "method") == other),
                        "damage",
                    ).max()
                )
                for other in PRIMARY_METHODS
            )
            trailing = span > 0.6 * widest
            axis_b.annotate(
                f"{METHOD_LABELS[method]}  {share:.2f}",
                (span, share * span),
                textcoords="offset points",
                xytext=(4, -1) if trailing else (-2, 5),
                fontsize=6.5,
                color=METHOD_COLORS[method],
                va="center" if trailing else "bottom",
                ha="left" if trailing else "right",
                zorder=6,
                path_effects=[
                    patheffects.Stroke(linewidth=2.0, foreground=SURFACE),
                    patheffects.Normal(),
                ],
            )
        axis_b.scatter(
            subset["damage"],
            subset["gain"],
            s=15,
            marker=METHOD_MARKERS[method],
            facecolor=METHOD_COLORS[method],
            edgecolor=SURFACE,
            linewidth=0.35,
            # More than half the cells sit near the origin, so keep the marks
            # light enough that the pile there reads as density.
            alpha=0.7,
            zorder=3,
        )
    # Room to the right of the longest ray for its tip label. The aspect stays
    # equal, so the box is wider than tall but the diagonal is still 45 degrees,
    # which is what makes "full recovery" readable as a position.
    axis_b.set_xlim(lo, edge)
    axis_b.set_ylim(lo, hi)
    axis_b.set_aspect("equal")
    axis_b.set_anchor("C")
    axis_b.set_xlabel("Missingness damage $D_r$", color=INK_MUTED, fontsize=8)
    axis_b.set_ylabel("Restoration gain $R_r$", color=INK_MUTED, fontsize=8)
    axis_b.set_title(
        "(b) the gain tracks the damage", fontsize=8.5, color=INK, pad=5
    )
    axis_b.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
    axis_b.set_axisbelow(True)
    for spine in ("top", "right"):
        axis_b.spines[spine].set_visible(False)


def _correlation(panel_b: pd.DataFrame, method: str) -> float:
    subset = _rows(panel_b, _column(panel_b, "method") == method)
    return float(_column(subset, "damage").corr(_column(subset, "gain")))


def _share(
    panel_b: pd.DataFrame, method: str, rng: np.random.Generator | None = None
) -> tuple[float, float, float]:
    """
    Fraction of the damage restoration returns, with a bootstrap interval.

    Least squares through the origin, which is the quantity the paper claims and
    the estimator settled on when the pooled `mean(R)/mean(D)` was retracted as a
    pooling artifact. Correlation answers a different question, whether `R` moves
    with `D`, and the two rank the methods differently.

    The fit weights each cell by `D^2`, so the more than half of all cells that
    sit at `|D| < 0.01` barely count. That is the point: a cell that lost nothing
    carries no information about what share comes back.
    """
    subset = _rows(panel_b, _column(panel_b, "method") == method)
    damage = _column(subset, "damage").to_numpy()
    gain = _column(subset, "gain").to_numpy()
    if not len(damage) or not float(damage @ damage):
        return (float("nan"), float("nan"), float("nan"))
    point = float(damage @ gain / (damage @ damage))
    rng = rng or np.random.default_rng(SEED)
    draws = rng.integers(0, len(damage), (BOOTSTRAP_DRAWS, len(damage)))
    shares = [
        float(damage[i] @ gain[i] / (damage[i] @ damage[i]))
        for i in draws
        if float(damage[i] @ damage[i])
    ]
    return (
        point,
        float(np.percentile(shares, 2.5)),
        float(np.percentile(shares, 97.5)),
    )


def _method_handles() -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=SURFACE,
            markeredgewidth=0.5,
            label=METHOD_LABELS[method],
        )
        for method in PRIMARY_METHODS
    ]


def plot_levels(
    levels: pd.DataFrame, output: Path, *, absolute: bool = False
) -> None:
    apply_paper_style()
    suffix = "_abs" if absolute else ""
    # Datasets ordered by the largest move any method makes on them.
    moves = levels.assign(
        move=(
            _column(levels, f"direct{suffix}")
            - _column(levels, f"generative{suffix}")
        ).abs()
    )
    ranked = cast(
        "pd.Series", moves.groupby("dataset")["move"].max()
    ).sort_values(ascending=False)
    datasets = [str(dataset) for dataset in ranked.index]
    # Four columns past six datasets, so eight fills a 2x4 exactly rather than
    # leaving a hole in a 3x3. Below that three columns keep the panels wide.
    columns = 4 if len(datasets) > 6 else min(3, len(datasets))
    rows = -(-len(datasets) // columns)
    # Gap units share y, since a gap whose panels autoscale independently is
    # not shared at all. Raw units cannot: the datasets sit at different levels
    # and mix accuracy with macro-F1, so one axis compresses every panel.
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(TEXT_WIDTH_IN, 1.35 + 1.55 * rows),
        squeeze=False,
        sharey=not absolute,
    )
    for index, dataset in enumerate(datasets):
        _draw_levels(
            axes[index // columns][index % columns],
            levels,
            dataset,
            absolute=absolute,
        )
    for index in range(len(datasets), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    height = 1.35 + 1.55 * rows
    # From the constant, so the label cannot drift from the cell shown.
    figure.supxlabel(
        f"{MECHANISM_LABELS[PANEL_MECHANISM]} missingness rate $p$",
        fontsize=8,
        y=LEGEND_STRIP_IN * 0.65 / height,
    )
    figure.supylabel(
        "Accuracy or macro-F1" if absolute else "Gap below ceiling",
        fontsize=8,
        x=0.015,
    )
    handles = [
        *_method_handles(),
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
        ),
    ]
    if absolute:
        handles.append(
            Line2D(
                [],
                [],
                marker="_",
                linestyle="none",
                markersize=7,
                markeredgecolor=INK_MUTED,
                markeredgewidth=1.2,
                label="Complete-data ceiling",
            )
        )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3 if absolute else 5,
        frameon=False,
        fontsize=6.5,
        labelcolor=INK_MUTED,
        columnspacing=1.2,
        handlelength=1.4,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.985,
        top=0.92,
        bottom=LEGEND_STRIP_IN / height,
        hspace=0.45,
        wspace=0.30,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)


def plot_law(law: pd.DataFrame, output: Path) -> None:
    apply_paper_style()
    # No legend box: four series are direct-labelled at the ray tips, where the
    # number each carries is the slope of the line it sits on.
    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 3.3))
    axis = figure.add_axes((0.10, 0.13, 0.88, 0.84))
    _draw_law(axis, law)
    axis.set_title("")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("extra/output/missing_data/summary/val"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "extra/output/missing_data/analysis_figures/main_summary.pdf"
        ),
    )
    parser.add_argument(
        "--law-output",
        type=Path,
        default=None,
        help=(
            "Where to write the gain-against-damage figure. Defaults to "
            "law.pdf beside --output."
        ),
    )
    parser.add_argument(
        "--absolute-output",
        type=Path,
        default=None,
        help=(
            "Where to write the same levels in raw metric units with each "
            "method's complete-data ceiling ticked. Defaults to "
            "<output>_absolute.pdf."
        ),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help=(
            "Write the per-cell damage and restoration gain behind the law "
            "figure. Defaults to <output>.cells.csv."
        ),
    )
    arguments = parser.parse_args()

    levels = collect_levels(arguments.summary_root)
    law = collect_panel_b(arguments.summary_root)
    if levels.empty or law.empty:
        message = "no cells collected"
        raise SystemExit(message)
    law_output = arguments.law_output or arguments.output.with_name("law.pdf")
    absolute_output = arguments.absolute_output or arguments.output.with_name(
        f"{arguments.output.stem}_absolute{arguments.output.suffix}"
    )
    plot_levels(levels, arguments.output)
    plot_levels(levels, absolute_output, absolute=True)
    plot_law(law, law_output)

    table = arguments.table or arguments.output.with_suffix(".cells.csv")
    law.to_csv(table, index=False)

    print(f"level rows: {len(levels)}   law cells: {len(law)}")
    print(
        levels.assign(closed=levels["direct"] - levels["generative"])
        .round(3)
        .to_string(index=False)
    )
    for method in PRIMARY_METHODS:
        subset = _rows(law, _column(law, "method") == method)
        share, low, high = _share(law, method)
        # The share over materially damaged cells alone. Where the two
        # disagree, the fit is carried by cells that lost nothing.
        material = _rows(subset, _column(subset, "damage") >= 0.01)
        damage = _column(material, "damage").to_numpy()
        gain = _column(material, "gain").to_numpy()
        restricted = (
            float(damage @ gain / (damage @ damage))
            if len(damage)
            else float("nan")
        )
        print(
            f"  {METHOD_LABELS[method]:18s} n={len(subset):3d} "
            f"r={_correlation(law, method):+.3f} "
            f"share={share:.3f} [{low:.3f}, {high:.3f}] "
            f"share|D>=0.01={restricted:.3f} (n={len(material)})"
        )
    print(
        f"wrote {arguments.output}, {absolute_output}, {law_output} "
        f"and {table}"
    )


if __name__ == "__main__":
    main()
