"""Render restricted-action training against generative restoration."""

# Writes a full-width level figure and a restoration-law figure for every
# induced missingness mechanism.

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
    DATASET_LABELS_SHORT,
    GRID,
    INDUCED_MECHANISMS,
    INK_MUTED,
    MECHANISM_LABELS,
    METHOD_COLORS,
    METHOD_FAMILIES,
    METHOD_LABELS,
    POLICY_TYPE_LINESTYLES,
    PRIMARY_METHODS,
    SURFACE,
    TEXT_WIDTH_IN,
    WEDGE,
    apply_paper_style,
    policy_type,
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
MAIN_MECHANISM = "mcar"
# 56% of cells carry damage of at least 0.01 at p = 0.7 against 35% at
# p = 0.3, where a third of the panel would be dumbbells of zero length.
MAIN_RATE = 0.7


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
    mechanism: str,
    rate: float,
    rng: np.random.Generator,
) -> dict[str, float] | None:
    """One method's direct and restored levels at one rate, in both units."""

    def cells(strategy: str) -> pd.DataFrame:
        return _rows(
            per_method,
            (_column(per_method, "strategy") == strategy)
            & (_column(per_method, "mechanism") == mechanism)
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
                for mechanism in INDUCED_MECHANISMS:
                    mechanism_rows = _rows(
                        per_method,
                        _column(per_method, "mechanism") == mechanism,
                    )
                    rates = sorted(
                        {
                            float(value)
                            for value in _column(mechanism_rows, "p")
                            if float(value) > 0.0
                        }
                    )
                    for rate in rates:
                        record = _level_row(
                            per_method,
                            metric,
                            gap_column,
                            mechanism,
                            rate,
                            rng,
                        )
                        if record is None:
                            continue
                        rows.append(
                            {
                                "dataset": dataset,
                                "method": method,
                                "mechanism": mechanism,
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
    axis: Axes,
    levels: pd.DataFrame,
    dataset: str,
    methods: list[str],
    *,
    x_limits: tuple[float, float] | None = None,
) -> None:
    """One dataset's panel: a dumbbell per method, methods down the y axis."""
    per_dataset = _rows(levels, _column(levels, "dataset") == dataset)
    for index, method in enumerate(methods):
        if policy_type(method) == "Myopic":
            axis.axhspan(
                index - 0.5, index + 0.5, color=WEDGE, linewidth=0, zorder=0
            )
    for index, method in enumerate(methods):
        record = _rows(
            per_dataset, _column(per_dataset, "method") == method
        ).iloc[0]
        color = METHOD_COLORS[method]
        for key in ("direct_abs", "generative_abs"):
            low, high = record[f"{key}_lo"], record[f"{key}_hi"]
            if np.isnan(low):
                continue
            axis.plot(
                [low, high],
                [index, index],
                color=color,
                linewidth=2.6,
                alpha=0.18,
                solid_capstyle="butt",
                zorder=1,
            )
        axis.plot(
            [record["direct_abs"], record["generative_abs"]],
            [index, index],
            color=color,
            linewidth=1.2,
            linestyle=POLICY_TYPE_LINESTYLES[policy_type(method)],
            zorder=2,
        )
        # Grey, because the ceiling is a reference rather than a third series.
        # It has to stay: DIME and ODIN take almost no damage on the datasets
        # where their ceiling is already the lowest of the nine, and without
        # this mark that floor effect reads as robustness.
        axis.plot(
            [record["ceiling_abs"]] * 2,
            [index - 0.32, index + 0.32],
            color=INK_MUTED,
            linewidth=1.0,
            zorder=3,
        )
        axis.scatter(
            [record["direct_abs"]],
            [index],
            s=17,
            facecolor=SURFACE,
            edgecolor=color,
            linewidth=0.9,
            zorder=4,
        )
        axis.scatter(
            [record["generative_abs"]],
            [index],
            s=20,
            color=color,
            zorder=5,
        )
    axis.set_ylim(len(methods) - 0.5, -0.5)
    axis.set_yticks(range(len(methods)))
    # Set on every panel, not only the first: the axes share y, so a bare
    # list on a later panel would clear the shared formatter for all of them.
    # sharey hides the inner columns' copies.
    axis.set_yticklabels(
        [METHOD_LABELS[method] for method in methods], fontsize=6
    )
    axis.tick_params(axis="y", length=0)
    if x_limits is not None:
        axis.set_xlim(*x_limits)
    axis.set_title(DATASET_LABELS_SHORT.get(dataset, dataset), fontsize=7.5)
    axis.grid(True, axis="x", color=GRID, linewidth=0.4, alpha=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(axis="x", labelsize=6)
    axis.locator_params(axis="x", nbins=4)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)


def _law_bounds(law: pd.DataFrame) -> tuple[float, float]:
    lo = min(law["damage"].min(), law["gain"].min()) - 0.03
    hi = max(law["damage"].max(), law["gain"].max()) + 0.03
    return lo, hi


def _draw_law(
    axis: Axes,
    cells: pd.DataFrame,
    method: str,
    bounds: tuple[float, float],
) -> None:
    """One method's panel: its cells, its fitted share, the two references."""
    lo, hi = bounds
    axis.fill_between(
        [0.0, hi], [0.0, 0.0], [0.0, hi], color=WEDGE, linewidth=0, zorder=0
    )
    axis.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=0.7, zorder=1)
    axis.axhline(0.0, color=INK_MUTED, linewidth=0.7, zorder=1)
    subset = _rows(cells, _column(cells, "method") == method)
    color = METHOD_COLORS[method]
    share, low, high = _share(cells, method)
    span = float(_column(subset, "damage").max()) if not subset.empty else 0.0
    if not np.isnan(share) and span > 0:
        axis.fill_between(
            [0.0, span],
            [0.0, low * span],
            [0.0, high * span],
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            [0.0, span],
            [0.0, share * span],
            color=color,
            linewidth=1.4,
            linestyle=POLICY_TYPE_LINESTYLES[policy_type(method)],
            solid_capstyle="round",
            path_effects=[
                patheffects.Stroke(linewidth=2.8, foreground=SURFACE),
                patheffects.Normal(),
            ],
            zorder=4,
        )
    axis.scatter(
        subset["damage"],
        subset["gain"],
        s=9,
        facecolor=color,
        edgecolor=SURFACE,
        linewidth=0.3,
        alpha=0.75,
        zorder=3,
    )
    material = int((_column(subset, "damage") >= 0.01).sum())
    axis.set_title(METHOD_LABELS[method], fontsize=7)
    # The count is the number that says whether the slope means anything: a
    # share fitted over six damaged cells is not the same claim as one over
    # eighteen.
    axis.annotate(
        f"{share:.2f}  ($n={material}$)",
        (0.05, 0.94),
        xycoords="axes fraction",
        fontsize=6,
        color=INK_MUTED,
        va="top",
    )
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    axis.set_aspect("equal")
    axis.grid(True, color=GRID, linewidth=0.4, alpha=0.6)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=6)
    axis.locator_params(nbins=4)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


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


def _mechanism_rows(frame: pd.DataFrame, mechanism: str) -> pd.DataFrame:
    selected = _rows(frame, _column(frame, "mechanism") == mechanism)
    if selected.empty:
        message = f"no cells for mechanism {mechanism}"
        raise ValueError(message)
    return selected


def _method_order(levels: pd.DataFrame) -> list[str]:
    """
    Families most damaged first, and inside a family $Q(s,a)$ above $Q(s,m,a)$.

    Ordering by damage rather than by policy type is what makes GDFS legible: a
    greedy method sits fourth, among the non-myopic ones, which the myopic band
    marks where it happens.
    """
    damaged = levels.assign(
        damage=_column(levels, "ceiling_abs") - _column(levels, "direct_abs"),
        family=_column(levels, "method").map(METHOD_FAMILIES),
    )
    ranked = cast(
        "pd.Series", damaged.groupby("family")["damage"].mean()
    ).sort_values(ascending=False)
    present = set(_column(levels, "method"))
    return [
        method
        for family in ranked.index
        for method in PRIMARY_METHODS
        if METHOD_FAMILIES[method] == family and method in present
    ]


def _dataset_order(levels: pd.DataFrame) -> list[str]:
    """Most damaged first, which is also the route-structure ordering."""
    damaged = levels.assign(
        damage=_column(levels, "ceiling_abs") - _column(levels, "direct_abs")
    )
    ranked = cast(
        "pd.Series", damaged.groupby("dataset")["damage"].mean()
    ).sort_values(ascending=False)
    return [str(dataset) for dataset in ranked.index]


def _absolute_limits(levels: pd.DataFrame) -> dict[str, tuple[float, float]]:
    columns = [
        "direct_abs_lo",
        "direct_abs_hi",
        "generative_abs_lo",
        "generative_abs_hi",
        "ceiling_abs",
    ]
    limits = {}
    for dataset in _column(levels, "dataset").unique():
        per_dataset = _rows(levels, _column(levels, "dataset") == dataset)
        values = per_dataset[columns].to_numpy(dtype=float)
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        padding = max(0.015, 0.09 * (high - low))
        limits[str(dataset)] = (low - padding, high + padding)
    return limits


def _level_legend() -> list[Line2D]:
    """
    Five entries, because the y axis already names every method.

    Identity moved to position, which is what freed colour to mean family and
    freed the legend to explain the two training views instead of listing nine
    series.
    """
    return [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            markerfacecolor=SURFACE,
            markeredgecolor=INK_MUTED,
            markeredgewidth=0.9,
            label="Restricted-action training",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.5,
            color=INK_MUTED,
            label="Generative restoration",
        ),
        Line2D(
            [],
            [],
            marker="|",
            linestyle="none",
            markersize=6,
            markeredgecolor=INK_MUTED,
            markeredgewidth=1.1,
            label="Complete-data ceiling",
        ),
        Line2D(
            [],
            [],
            color=INK_MUTED,
            linewidth=1.2,
            linestyle=POLICY_TYPE_LINESTYLES["Myopic"],
            label="Myopic",
        ),
        Line2D(
            [],
            [],
            color=INK_MUTED,
            linewidth=1.2,
            linestyle=POLICY_TYPE_LINESTYLES["Non-myopic"],
            label="Non-myopic",
        ),
    ]


def plot_levels(
    levels: pd.DataFrame,
    output: Path,
    *,
    mechanism: str,
    rate: float,
    dataset_order: list[str] | None = None,
    method_order: list[str] | None = None,
) -> None:
    apply_paper_style()
    per_mechanism = _mechanism_rows(levels, mechanism)
    frame = _rows(per_mechanism, _column(per_mechanism, "p") == rate)
    if frame.empty:
        message = f"no cells for {mechanism} at p={rate}"
        raise ValueError(message)
    datasets = dataset_order or _dataset_order(frame)
    methods = method_order or _method_order(frame)
    limits = _absolute_limits(frame)
    columns = 4
    rows = -(-len(datasets) // columns)
    # Method identity is on the y axis; this strip carries the two training
    # views, ceiling, and policy-type conventions in two compact rows.
    strip = 0.82
    height = strip + 1.75 * rows
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(TEXT_WIDTH_IN, height),
        squeeze=False,
        sharey=True,
    )
    for index, dataset in enumerate(datasets):
        row, column = divmod(index, columns)
        _draw_levels(
            axes[row][column],
            frame,
            dataset,
            methods,
            x_limits=limits.get(dataset),
        )
    for index in range(len(datasets), rows * columns):
        row, column = divmod(index, columns)
        axes[row][column].set_visible(False)

    figure.supxlabel(
        "Accuracy or macro-F1",
        fontsize=8,
        y=0.48 / height,
    )
    figure.legend(
        handles=_level_legend(),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.5,
        labelcolor=INK_MUTED,
        columnspacing=1.0,
        handlelength=1.6,
        bbox_to_anchor=(0.5, 0.012),
    )
    figure.subplots_adjust(
        left=0.195,
        right=0.985,
        top=0.93,
        bottom=strip / height,
        hspace=0.34,
        wspace=0.14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(figure)


def plot_law(
    law: pd.DataFrame,
    output: Path,
    *,
    mechanism: str,
    bounds: tuple[float, float] | None = None,
    method_order: list[str] | None = None,
) -> None:
    apply_paper_style()
    per_mechanism = _mechanism_rows(law, mechanism)
    methods = method_order or [
        method
        for method in PRIMARY_METHODS
        if (_column(per_mechanism, "method") == method).any()
    ]
    columns = 3
    rows = -(-len(methods) // columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(TEXT_WIDTH_IN, 1.72 * rows + 0.55),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for index, method in enumerate(methods):
        row, column = divmod(index, columns)
        _draw_law(
            axes[row][column],
            per_mechanism,
            method,
            bounds or _law_bounds(per_mechanism),
        )
    for index in range(len(methods), rows * columns):
        row, column = divmod(index, columns)
        axes[row][column].set_visible(False)
    figure.supxlabel("Missingness damage $D_r$", fontsize=8, y=0.02)
    figure.supylabel("Restoration gain $R_r$", fontsize=8, x=0.015)
    figure.subplots_adjust(
        left=0.10, right=0.99, top=0.955, bottom=0.10, hspace=0.30, wspace=0.10
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    figure.savefig(output.with_suffix(".png"), dpi=200)
    plt.close(figure)


def plot_mechanism_figures(
    levels: pd.DataFrame,
    law: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Render a full-width level/law pair for every mechanism."""
    main = _mechanism_rows(levels, MAIN_MECHANISM)
    main = _rows(main, _column(main, "p") == MAIN_RATE)
    order = _dataset_order(main)
    methods = _method_order(main)
    bounds = _law_bounds(law)
    outputs = []
    for mechanism in INDUCED_MECHANISMS:
        levels_output = output_dir / f"main_summary_absolute_{mechanism}.pdf"
        law_output = output_dir / f"law_{mechanism}.pdf"
        plot_levels(
            levels,
            levels_output,
            mechanism=mechanism,
            rate=MAIN_RATE,
            dataset_order=order,
            method_order=methods,
        )
        plot_law(
            law,
            law_output,
            mechanism=mechanism,
            bounds=bounds,
            method_order=methods,
        )
        outputs.extend([levels_output, law_output])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("extra/output/missing_data/summary/val"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extra/output/missing_data/analysis_figures"),
        help="Where the per-mechanism level and law figures are written.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=None,
        help=(
            "Write the per-cell damage and restoration gain behind the law "
            "figure. Defaults to main_summary.cells.csv beside --output-dir."
        ),
    )
    arguments = parser.parse_args()

    levels = collect_levels(arguments.summary_root)
    law = collect_panel_b(arguments.summary_root)
    if levels.empty or law.empty:
        message = "no cells collected"
        raise SystemExit(message)
    outputs = plot_mechanism_figures(levels, law, arguments.output_dir)

    table = arguments.table or arguments.output_dir / "main_summary.cells.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    law.to_csv(table, index=False)

    print(f"level rows: {len(levels)}   law cells: {len(law)}")
    for mechanism in INDUCED_MECHANISMS:
        per_mechanism = _mechanism_rows(law, mechanism)
        print(f"{MECHANISM_LABELS[mechanism]}:")
        for method in PRIMARY_METHODS:
            subset = _rows(
                per_mechanism, _column(per_mechanism, "method") == method
            )
            share, low, high = _share(per_mechanism, method)
            material = _rows(subset, _column(subset, "damage") >= 0.01)
            print(
                f"  {METHOD_LABELS[method]:20s} n={len(subset):3d} "
                f"r={_correlation(per_mechanism, method):+.3f} "
                f"share={share:.3f} [{low:.3f}, {high:.3f}] "
                f"material={len(material)}"
            )
    print(
        "wrote " + ", ".join(str(path) for path in outputs) + f" and {table}"
    )


if __name__ == "__main__":
    main()
