"""Render the paper's main figure, direct learning against generative restoration."""

# Panel (a) is what a practitioner gets, one dumbbell per dataset and method in
# that dataset's own primary metric with the complete-data ceiling ticked. The
# comparison is within a row, so mixing accuracy and macro-F1 across rows is
# sound. Panel (b) is why, the restoration gain against the missingness damage
# over every dataset, method, mechanism and rate.

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
from matplotlib.lines import Line2D

from afabench.plotting.methods import (
    METHOD_COLORS,
    METHOD_LABELS,
    PRIMARY_METHODS,
    TEXT_WIDTH_IN,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
WEDGE = "#f0efec"
SURFACE = "#ffffff"

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
    ],
}

DATASET_LABELS = {
    "cube_nm": "CUBE-NM",
    "cube": "CUBE",
    "cube_nonuniform_costs": "CUBE non-uniform cost",
    "heart_disease": "Heart disease",
    "actg": "ACTG175",
    "diabetes": "Diabetes",
    "nhanes_mortality": "NHANES mortality",
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


def collect_levels(summary_root: Path) -> pd.DataFrame:
    """
    Distance below each method's own complete-data ceiling, per rate.

    Reported as a gap rather than a raw score so every dataset shares a scale on
    which zero means the same thing. In raw units ACTG175 spans 0.58 to 0.85 and
    Diabetes 0.60 to 0.69, so one axis compresses both and an 0.02 move reads
    differently per row.
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
                rates = sorted(
                    {
                        float(value)
                        for value in _column(per_method, "p")
                        if float(value) > 0.0
                    }
                )
                for rate in rates:

                    def gaps(
                        strategy: str,
                        per_method: pd.DataFrame = per_method,
                        gap_column: str = gap_column,
                        rate: float = rate,
                    ) -> npt.NDArray[np.float64]:
                        cells = _rows(
                            per_method,
                            (_column(per_method, "strategy") == strategy)
                            & (
                                _column(per_method, "mechanism")
                                == PANEL_MECHANISM
                            )
                            & (_column(per_method, "p") == rate),
                        )
                        return -_column(cells, gap_column).dropna().to_numpy()

                    direct, generative = gaps(DIRECT), gaps(GENERATIVE)
                    if not len(direct) or not len(generative):
                        continue
                    direct_lo, direct_hi = _interval(direct, rng)
                    generative_lo, generative_hi = _interval(generative, rng)
                    rows.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "p": rate,
                            "metric": metric,
                            "direct": float(direct.mean()),
                            "direct_lo": direct_lo,
                            "direct_hi": direct_hi,
                            "generative": float(generative.mean()),
                            "generative_lo": generative_lo,
                            "generative_hi": generative_hi,
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


def _draw_levels(axis: Axes, levels: pd.DataFrame, dataset: str) -> None:
    """One dataset's panel: a dumbbell per method at each missingness rate."""
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
            for key in ("direct", "generative"):
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
                [record["direct"], record["generative"]],
                color=color,
                linewidth=1.3,
                solid_capstyle="round",
                zorder=2,
            )
            axis.scatter(
                [x],
                [record["direct"]],
                s=13,
                facecolor=SURFACE,
                edgecolor=color,
                linewidth=1.0,
                zorder=3,
            )
            axis.scatter(
                [x], [record["generative"]], s=16, color=color, zorder=4
            )
    # Zero is the complete-data ceiling for every dataset, which is the point of
    # plotting a gap: one scale on which zero means the same thing.
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
        (hi, 0.0),
        textcoords="offset points",
        xytext=(-3, -15),
        fontsize=6.5,
        color=INK_MUTED,
        ha="right",
        va="top",
    )
    # Colour alone carries the method here. Mechanism moved to the
    # identification figure, which took this panel from 16 classes to 4.
    for method in PRIMARY_METHODS:
        subset = _rows(panel_b, _column(panel_b, "method") == method)
        if subset.empty:
            continue
        axis_b.scatter(
            subset["damage"],
            subset["gain"],
            s=17,
            marker="o",
            facecolor=METHOD_COLORS[method],
            edgecolor=SURFACE,
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )
    axis_b.set_xlim(lo, hi)
    axis_b.set_ylim(lo, hi)
    axis_b.set_aspect("equal")
    # Square data box, so anchor it to the top of its cell to keep the two
    # panel titles on one line.
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


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 8,
            "text.color": INK,
            "axes.labelcolor": INK_MUTED,
            "axes.edgecolor": GRID,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
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


def plot_levels(levels: pd.DataFrame, output: Path) -> None:
    _style()
    # Datasets ordered by the largest move any method makes on them.
    moves = levels.assign(
        move=(_column(levels, "direct") - _column(levels, "generative")).abs()
    )
    ranked = cast(
        "pd.Series", moves.groupby("dataset")["move"].max()
    ).sort_values(ascending=False)
    datasets = [str(dataset) for dataset in ranked.index]
    columns = min(3, len(datasets))
    rows = -(-len(datasets) // columns)
    # Shared y, because a gap scale whose panels autoscale independently is no
    # longer shared and the whole reason for plotting a gap is lost. That NHANES
    # then looks flat beside ACTG175 is the finding, not a defect.
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(TEXT_WIDTH_IN, 1.35 + 1.55 * rows),
        squeeze=False,
        sharey=True,
    )
    for index, dataset in enumerate(datasets):
        _draw_levels(axes[index // columns][index % columns], levels, dataset)
    for index in range(len(datasets), rows * columns):
        axes[index // columns][index % columns].set_visible(False)

    figure.supxlabel("Training missingness rate", fontsize=8, y=0.215)
    figure.supylabel("Gap below complete-data ceiling", fontsize=8, x=0.015)
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
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=6.5,
        labelcolor=INK_MUTED,
        columnspacing=1.2,
        handlelength=1.4,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        left=0.11, right=0.985, top=0.92, bottom=0.32, hspace=0.45, wspace=0.30
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)


def plot_law(law: pd.DataFrame, output: Path) -> None:
    _style()
    # The data box is square, since a 45 degree diagonal is what makes "full
    # recovery" read as a position. A square box cannot fill the text width, so
    # the legend takes the space beside it and carries each correlation.
    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 3.1))
    axis = figure.add_axes((0.11, 0.13, 0.55, 0.84))
    _draw_law(axis, law)
    axis.set_title("")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            markerfacecolor=METHOD_COLORS[method],
            markeredgecolor=SURFACE,
            markeredgewidth=0.5,
            label=(
                f"{METHOD_LABELS[method]}\n"
                f"    $r={_correlation(law, method):.2f}$"
            ),
        )
        for method in PRIMARY_METHODS
    ]
    figure.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=7,
        labelcolor=INK_MUTED,
        labelspacing=1.1,
        handletextpad=0.6,
        bbox_to_anchor=(0.68, 0.55),
    )
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
    plot_levels(levels, arguments.output)
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
        print(
            f"  {METHOD_LABELS[method]:5s} n={len(subset):3d} "
            f"r={_correlation(law, method):+.3f}"
        )
    print(f"wrote {arguments.output}, {law_output} and {table}")


if __name__ == "__main__":
    main()
