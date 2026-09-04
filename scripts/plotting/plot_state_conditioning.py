"""Compare mask-aware and aliasing Q states within JAFA, OL, and ODIN."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from afabench.plotting.methods import (
    DATASET_LABELS_SHORT,
    FAMILY_COLORS,
    GRID,
    INK_MUTED,
    LEGEND_STRIP_IN,
    SURFACE,
    TEXT_WIDTH_IN,
    apply_paper_style,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


ACCURACY_DATASETS = {"cube", "cube_nm", "cube_nonuniform_costs"}
RESTRICTED = "restricted"
GENERATIVE = "pvae_label_conditioned"
FAMILY_METHODS = {
    "JAFA": ("jafa", "jafa_full_state"),
    "OL": ("ol_with_mask", "ol_full_state"),
    "ODIN": ("odin_model_free", "odin_model_free_full_state"),
}
FAMILY_KEYS = {"JAFA": "jafa", "OL": "ol", "ODIN": "odin"}
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
DATASET_MARKERS = {
    "cube": "o",
    "cube_nm": "s",
    "cube_nonuniform_costs": "D",
    "heart_disease": "P",
    "actg": "^",
    "diabetes": "v",
    "nhanes_mortality": "X",
    "miniboone": "*",
}
RATE_SIZES = {0.3: 18.0, 0.5: 28.0, 0.7: 40.0}


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast("pd.Series", frame[name])


def _rows(frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    return cast("pd.DataFrame", frame[mask])


def primary_metric(dataset: str) -> str:
    return "accuracy" if dataset in ACCURACY_DATASETS else "f_score"


def _paired_state_difference(
    frame: pd.DataFrame,
    dataset: str,
    short_method: str,
    full_method: str,
    mechanism: str,
) -> pd.DataFrame:
    """Return full-minus-short differences at identical scientific cells."""
    metric = primary_metric(dataset)
    selected = _rows(
        frame,
        (_column(frame, "dataset") == dataset)
        & _column(frame, "method").isin([short_method, full_method])
        & _column(frame, "strategy").isin(
            ["complete", RESTRICTED, GENERATIVE]
        ),
    ).copy()
    if selected.empty:
        return pd.DataFrame()
    largest_budget = float(_column(selected, "eval_hard_budget").max())
    selected = _rows(
        selected, _column(selected, "eval_hard_budget") == largest_budget
    )
    key = [
        "dataset",
        "mechanism",
        "p",
        "strategy",
        "instance",
        "train_hard_budget",
        "eval_hard_budget",
    ]
    if selected.duplicated([*key, "method"]).any():
        message = f"duplicate state-conditioning cells for {dataset}"
        raise ValueError(message)
    wide = selected.pivot_table(
        index=key,
        columns="method",
        values=metric,
        aggfunc="first",
    )
    if short_method not in wide or full_method not in wide:
        return pd.DataFrame()
    wide = wide.dropna(subset=[short_method, full_method]).reset_index()
    wide["state_difference"] = wide[full_method] - wide[short_method]

    complete = _rows(wide, _column(wide, "strategy") == "complete")[
        [
            "dataset",
            "instance",
            "train_hard_budget",
            "eval_hard_budget",
            "state_difference",
        ]
    ].rename(columns={"state_difference": "complete_difference"})  # pyright: ignore[reportCallIssue]
    missing = _rows(
        wide,
        (_column(wide, "mechanism") == mechanism)
        & _column(wide, "strategy").isin([RESTRICTED, GENERATIVE]),
    )
    paired = missing.merge(
        complete,
        on=[
            "dataset",
            "instance",
            "train_hard_budget",
            "eval_hard_budget",
        ],
        how="inner",
        validate="many_to_one",
    )
    paired["adjusted_difference"] = (
        paired["state_difference"] - paired["complete_difference"]
    )
    return paired


def _summary_rows(
    paired: pd.DataFrame,
    family: str,
    mechanism: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped = cast("Any", paired.groupby(["dataset", "p"]))
    for (dataset, rate), cell in grouped:
        values: dict[str, float] = {}
        counts: dict[str, int] = {}
        for strategy in (RESTRICTED, GENERATIVE):
            arm = _rows(cell, _column(cell, "strategy") == strategy)
            values[strategy] = float(
                _column(arm, "adjusted_difference").mean()
            )
            counts[strategy] = len(arm)
        rows.append(
            {
                "family": family,
                "dataset": dataset,
                "mechanism": mechanism,
                "p": float(rate),
                "restricted": values[RESTRICTED],
                "restored": values[GENERATIVE],
                "n_restricted": counts[RESTRICTED],
                "n_restored": counts[GENERATIVE],
            }
        )
    return rows


def collect(summary_root: Path, mechanism: str = "mcar") -> pd.DataFrame:
    """Collect instance-paired, complete-adjusted contrasts for three families."""
    rows: list[dict[str, object]] = []
    for namespace, datasets in SOURCES.items():
        path = summary_root / namespace / "instance_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for family, (short_method, full_method) in FAMILY_METHODS.items():
            for dataset in datasets:
                paired = _paired_state_difference(
                    frame,
                    dataset,
                    short_method,
                    full_method,
                    mechanism,
                )
                if paired.empty:
                    continue
                rows.extend(_summary_rows(paired, family, mechanism))
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    expected = {
        (family, dataset, rate)
        for family in FAMILY_METHODS
        for datasets in SOURCES.values()
        for dataset in datasets
        for rate in (0.3, 0.5, 0.7)
    }
    observed = set(
        zip(result["family"], result["dataset"], result["p"], strict=True)
    )
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        message = (
            "state-conditioning coverage mismatch: "
            f"missing={missing}; extra={extra}"
        )
        raise ValueError(message)
    if not (result[["n_restricted", "n_restored"]] == 5).all().all():
        message = "state-conditioning cells must contain five paired instances"
        raise ValueError(message)
    return result.sort_values(["family", "dataset", "p"]).reset_index(
        drop=True
    )


def _draw(
    axis: Axes,
    frame: pd.DataFrame,
    family: str,
    limits: tuple[float, float],
) -> None:
    lo, hi = limits
    axis.axhline(0.0, color=GRID, linewidth=0.8, zorder=0)
    axis.axvline(0.0, color=GRID, linewidth=0.8, zorder=0)
    axis.plot(
        [lo, hi],
        [lo, hi],
        color=INK_MUTED,
        linestyle="--",
        linewidth=0.8,
    )
    color = FAMILY_COLORS[FAMILY_KEYS[family]]
    for raw_row in frame.itertuples(index=False):
        row = cast("Any", raw_row)
        axis.scatter(
            row.restricted,
            row.restored,
            marker=DATASET_MARKERS[row.dataset],
            s=RATE_SIZES[float(row.p)],
            facecolor=color,
            edgecolor=SURFACE,
            linewidth=0.55,
            alpha=0.88,
            zorder=2,
        )
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(family, fontsize=8)
    axis.tick_params(labelsize=6.5)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)


def plot(frame: pd.DataFrame, output: Path) -> None:
    apply_paper_style()
    values = pd.concat([frame["restricted"], frame["restored"]])
    extent = max(abs(float(values.min())), abs(float(values.max()))) * 1.08
    limits = (-extent, extent)
    height = 2.75
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(TEXT_WIDTH_IN, height),
        sharex=True,
        sharey=True,
    )
    for axis, family in zip(axes, FAMILY_METHODS, strict=True):
        _draw(
            axis,
            _rows(frame, _column(frame, "family") == family),
            family,
            limits,
        )
    figure.supxlabel(
        "Adjusted contrast under restricted-action training",
        fontsize=8,
        y=LEGEND_STRIP_IN * 0.55 / height,
    )
    figure.supylabel(
        "Adjusted contrast after generative restoration",
        fontsize=8,
        x=0.01,
    )
    dataset_handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            markersize=4.2,
            markerfacecolor=INK_MUTED,
            markeredgecolor=SURFACE,
            label=DATASET_LABELS_SHORT[dataset],
        )
        for dataset, marker in DATASET_MARKERS.items()
    ]
    rate_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=(size**0.5) * 0.72,
            markerfacecolor=INK_MUTED,
            markeredgecolor=SURFACE,
            label=f"$p={rate:g}$",
        )
        for rate, size in RATE_SIZES.items()
    ]
    dataset_legend = figure.legend(
        handles=dataset_handles,
        loc="lower center",
        ncol=8,
        frameon=False,
        fontsize=5.7,
        labelcolor=INK_MUTED,
        columnspacing=0.65,
        handletextpad=0.25,
        bbox_to_anchor=(0.5, 0.065),
    )
    figure.add_artist(dataset_legend)
    figure.legend(
        handles=rate_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=6.0,
        labelcolor=INK_MUTED,
        columnspacing=1.1,
        handletextpad=0.25,
        bbox_to_anchor=(0.5, 0.002),
    )
    figure.subplots_adjust(
        left=0.105,
        right=0.99,
        top=0.91,
        bottom=LEGEND_STRIP_IN / height,
        wspace=0.14,
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
    parser.add_argument("--mechanism", default="mcar")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "extra/output/paper/experiments/results/state_conditioning.pdf"
        ),
    )
    parser.add_argument("--table", type=Path)
    arguments = parser.parse_args()

    frame = collect(arguments.summary_root, arguments.mechanism)
    if frame.empty:
        message = "no cells collected"
        raise SystemExit(message)
    plot(frame, arguments.output)
    table = arguments.table or arguments.output.with_suffix(".csv")
    frame.to_csv(table, index=False)
    summary = frame.groupby("family").agg(
        restricted=("restricted", "mean"),
        restored=("restored", "mean"),
    )
    print(summary.round(4).to_string())
    print(f"wrote {arguments.output} and {table}")


if __name__ == "__main__":
    main()
