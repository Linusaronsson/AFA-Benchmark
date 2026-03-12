# pyright: reportCallIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from afabench.eval.plotting_config import METHOD_NAME_MAPPING

if TYPE_CHECKING:
    from numpy.typing import NDArray

RISK_METRIC_KEYS = (
    "risky_episode_rate",
    "risky_accuracy",
    "risky_rescue_rate",
    "risky_unsafe_stop_rate",
)
SHIELD_METRIC_KEYS = (
    "proposed_stop_rate",
    "rejected_stop_rate",
    "forced_continue_rate",
)

DEFAULT_FORMATS = ("pdf", "svg")
DARK2_COLORS = (
    "#1B9E77",
    "#D95F02",
    "#7570B3",
    "#E7298A",
    "#66A61E",
    "#E6AB02",
    "#A6761D",
)
TRAIN_INITIALIZER_LABELS = {
    "cold": "full train support",
    "cube_nm_ar": "rescue-censored train",
    "mcar_p03": "MCAR 30%",
    "mar_p03": "MAR 30%",
    "mnar_logistic_p03": "MNAR logistic 30%",
}
TRAIN_INITIALIZER_LINESTYLES = {
    "cold": "-",
    "cube_nm_ar": "--",
    "mcar_p03": ":",
    "mar_p03": "-.",
    "mnar_logistic_p03": (0, (3, 1, 1, 1)),
}
METHOD_STYLES: dict[str, dict[str, str]] = {
    "gadgil2023": {
        "label": METHOD_NAME_MAPPING.get("gadgil2023", "gadgil2023"),
        "color": DARK2_COLORS[0],
        "linestyle": "-",
        "marker": "o",
    },
    "gadgil2023_ipw_feature_marginal": {
        "label": f"{METHOD_NAME_MAPPING.get('gadgil2023', 'gadgil2023')} + IPW",
        "color": DARK2_COLORS[1],
        "linestyle": "--",
        "marker": "s",
    },
    "aaco_full": {
        "label": "AACO",
        "color": DARK2_COLORS[2],
        "linestyle": "-",
        "marker": "^",
    },
    "aaco_zero_fill": {
        "label": "AACO + zero-fill",
        "color": DARK2_COLORS[3],
        "linestyle": "--",
        "marker": "D",
    },
    "aaco_mask_aware": {
        "label": "AACO + mask-aware",
        "color": DARK2_COLORS[4],
        "linestyle": "-.",
        "marker": "P",
    },
    "cube_nm_ar_oracle": {
        "label": "CUBE-NM-AR oracle",
        "color": "#111111",
        "linestyle": ":",
        "marker": "*",
    },
    "odin_model_based": {
        "label": METHOD_NAME_MAPPING.get(
            "odin_model_based", "odin_model_based"
        ),
        "color": DARK2_COLORS[5],
        "linestyle": "-",
        "marker": "X",
    },
    "odin_model_free": {
        "label": METHOD_NAME_MAPPING.get("odin_model_free", "odin_model_free"),
        "color": DARK2_COLORS[6],
        "linestyle": "--",
        "marker": "v",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize and plot CUBE-NM-AR evaluation logs for "
            "risky-rescue train-support experiments."
        )
    )
    parser.add_argument("eval_results_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", default="cube_nm_ar")
    parser.add_argument(
        "--budget-mode",
        choices=["hard", "soft"],
        default="hard",
    )
    parser.add_argument("--train-initializer", default=None)
    parser.add_argument(
        "--train-initializers",
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--eval-initializers",
        nargs="+",
        default=["cold"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "gadgil2023",
            "gadgil2023_ipw_feature_marginal",
            "aaco_full",
            "aaco_zero_fill",
            "aaco_mask_aware",
        ],
    )
    parser.add_argument("--n-contexts", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
    )
    return parser.parse_args()


def add_episode_id(df: pd.DataFrame) -> pd.DataFrame:
    starts = df["prev_selections_performed"].astype(str).eq("[]")
    episode_idx = starts.groupby(df["idx"]).cumsum() - 1
    out = df.copy()
    out["episode_id"] = df["idx"].astype(str) + ":" + episode_idx.astype(str)
    return out


def _parse_budget_value(raw_value: str) -> float | None:
    if raw_value == "null":
        return None
    return float(raw_value)


def _parse_eval_path_metadata(
    path: Path,
    *,
    pattern: re.Pattern[str],
) -> dict[str, int | float | None] | None:
    match = pattern.search(str(path))
    if match is None:
        return None
    (
        instance_idx,
        train_seed,
        train_hard_budget,
        train_soft_budget_param,
        eval_seed,
        eval_hard_budget,
        eval_soft_budget_param,
    ) = match.groups()
    return {
        "instance_idx": int(instance_idx),
        "train_seed": int(train_seed),
        "train_hard_budget": _parse_budget_value(train_hard_budget),
        "train_soft_budget_param": _parse_budget_value(
            train_soft_budget_param
        ),
        "eval_seed": int(eval_seed),
        "eval_hard_budget": _parse_budget_value(eval_hard_budget),
        "eval_soft_budget_param": _parse_budget_value(eval_soft_budget_param),
    }


def _matches_budget_mode(
    budget_mode: str,
    metadata: dict[str, int | float | None],
) -> bool:
    if budget_mode == "hard":
        return metadata["eval_hard_budget"] is not None
    return metadata["eval_soft_budget_param"] is not None


def _load_metric_summary(
    path: Path,
    metric_keys: tuple[str, ...],
) -> dict[str, float]:
    summary = {key: float("nan") for key in metric_keys}
    if not path.exists():
        return summary
    loaded_summary = json.loads(path.read_text(encoding="utf-8"))
    for key in metric_keys:
        if key in loaded_summary:
            summary[key] = float(loaded_summary[key])
    return summary


def _summarize_eval_run(
    eval_df: pd.DataFrame,
    *,
    rescue_action: int,
) -> pd.DataFrame:
    eval_df = add_episode_id(eval_df)
    final = eval_df.groupby("episode_id", as_index=False).tail(1).copy()
    first = eval_df.groupby("episode_id", as_index=False).head(1).copy()
    action_flags = eval_df.groupby("episode_id").agg(
        context_acquired=(
            "action_performed",
            lambda x: (x == 1).any(),
        ),
        rescue_acquired=(
            "action_performed",
            lambda x: (x == rescue_action).any(),
        ),
    )
    merged = final.merge(
        first[["episode_id", "action_performed"]].rename(
            columns={"action_performed": "first_action"}
        ),
        on="episode_id",
        how="left",
    ).merge(action_flags, on="episode_id", how="left")
    pred_col = (
        "external_predicted_class"
        if "external_predicted_class" in merged.columns
        and bool(merged["external_predicted_class"].notna().any())
        else "builtin_predicted_class"
    )
    merged["correct"] = (merged[pred_col] == merged["true_class"]).astype(
        float
    )
    return merged


def collect_episode_rows(
    root: Path,
    *,
    budget_mode: str,
    train_initializers: list[str],
    eval_initializers: list[str],
    dataset: str,
    methods: list[str],
    rescue_action: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pattern = re.compile(
        r"instance_idx-(\d+).+?"
        r"train_seed-(\d+)\+train_hard_budget-([^+/]+)\+train_soft_budget_param-([^/]+)/"
        r"eval_seed-(\d+)\+eval_hard_budget-([^+/]+)\+eval_soft_budget_param-([^/]+)/"
        r"eval_data\.csv$"
    )

    for train_initializer in train_initializers:
        for eval_initializer in eval_initializers:
            relative_init_root = Path(
                "train_initializer-"
                f"{train_initializer}+eval_initializer-{eval_initializer}"
            )
            init_root = (
                root
                if root.name == relative_init_root.name
                else root / relative_init_root
            )
            for method in methods:
                for path in init_root.glob(
                    f"{method}/dataset-{dataset}+instance_idx-*/**/eval_data.csv"
                ):
                    metadata = _parse_eval_path_metadata(
                        path,
                        pattern=pattern,
                    )
                    if metadata is None or not _matches_budget_mode(
                        budget_mode,
                        metadata,
                    ):
                        continue
                    merged = _summarize_eval_run(
                        pd.read_csv(path),
                        rescue_action=rescue_action,
                    )
                    risk_summary = _load_metric_summary(
                        path.with_name("cube_nm_ar_risk_summary.json"),
                        RISK_METRIC_KEYS,
                    )
                    shield_summary = _load_metric_summary(
                        path.with_name("stop_shield_summary.json"),
                        SHIELD_METRIC_KEYS,
                    )

                    rows.append(
                        {
                            "train_initializer": train_initializer,
                            "eval_initializer": eval_initializer,
                            "method": method,
                            **metadata,
                            "n_episodes": len(merged),
                            "accuracy": merged["correct"].mean(),
                            "mean_cost": merged["accumulated_cost"].mean(),
                            "forced_stop_rate": merged["forced_stop"]
                            .astype(float)
                            .mean(),
                            "context_episode_rate": merged["context_acquired"]
                            .astype(float)
                            .mean(),
                            "rescue_episode_rate": merged["rescue_acquired"]
                            .astype(float)
                            .mean(),
                            "first_action_context_rate": (
                                merged["first_action"] == 1
                            ).mean(),
                            **risk_summary,
                            **shield_summary,
                        }
                    )

    return pd.DataFrame(rows)


def _budget_column(budget_mode: str) -> str:
    if budget_mode == "hard":
        return "eval_hard_budget"
    return "eval_soft_budget_param"


def _budget_label(budget_mode: str) -> str:
    if budget_mode == "hard":
        return "Evaluation hard budget"
    return "Evaluation soft budget"


def aggregate_runs(summary: pd.DataFrame, *, budget_mode: str) -> pd.DataFrame:
    budget_column = _budget_column(budget_mode)
    aggregated = (
        summary.groupby(
            [
                "train_initializer",
                "eval_initializer",
                "method",
                budget_column,
            ],
            as_index=False,
        )
        .agg(
            runs=("instance_idx", "nunique"),
            episodes=("n_episodes", "sum"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            mean_cost_mean=("mean_cost", "mean"),
            mean_cost_std=("mean_cost", "std"),
            forced_stop_rate_mean=("forced_stop_rate", "mean"),
            forced_stop_rate_std=("forced_stop_rate", "std"),
            risky_episode_rate_mean=("risky_episode_rate", "mean"),
            risky_episode_rate_std=("risky_episode_rate", "std"),
            risky_accuracy_mean=("risky_accuracy", "mean"),
            risky_accuracy_std=("risky_accuracy", "std"),
            risky_rescue_rate_mean=("risky_rescue_rate", "mean"),
            risky_rescue_rate_std=("risky_rescue_rate", "std"),
            risky_unsafe_stop_rate_mean=("risky_unsafe_stop_rate", "mean"),
            risky_unsafe_stop_rate_std=("risky_unsafe_stop_rate", "std"),
            context_episode_rate_mean=("context_episode_rate", "mean"),
            context_episode_rate_std=("context_episode_rate", "std"),
            rescue_episode_rate_mean=("rescue_episode_rate", "mean"),
            rescue_episode_rate_std=("rescue_episode_rate", "std"),
            first_action_context_rate_mean=(
                "first_action_context_rate",
                "mean",
            ),
            first_action_context_rate_std=("first_action_context_rate", "std"),
            proposed_stop_rate_mean=("proposed_stop_rate", "mean"),
            proposed_stop_rate_std=("proposed_stop_rate", "std"),
            rejected_stop_rate_mean=("rejected_stop_rate", "mean"),
            rejected_stop_rate_std=("rejected_stop_rate", "std"),
            forced_continue_rate_mean=("forced_continue_rate", "mean"),
            forced_continue_rate_std=("forced_continue_rate", "std"),
        )
        .sort_values(
            [
                "method",
                "train_initializer",
                "eval_initializer",
                budget_column,
            ]
        )
    )
    std_columns = [
        column for column in aggregated.columns if column.endswith("_std")
    ]
    aggregated[std_columns] = aggregated[std_columns].fillna(0.0)
    return aggregated


def apply_paper_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "legend.handlelength": 2.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.transparent": False,
            "savefig.bbox": "tight",
            "grid.color": "#D0D0D0",
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.2,
            "lines.markersize": 6.5,
        }
    )


def _method_label(method: str) -> str:
    return METHOD_STYLES.get(method, {}).get("label", method)


def _method_style(method: str) -> dict[str, str]:
    return METHOD_STYLES.get(
        method,
        {
            "label": method,
            "color": "#4C4C4C",
            "linestyle": "-",
            "marker": "o",
        },
    )


def _train_initializer_label(train_initializer: str) -> str:
    return TRAIN_INITIALIZER_LABELS.get(
        train_initializer,
        train_initializer.replace("_", " "),
    )


def _train_initializer_linestyle(
    train_initializer: str,
    initializer_order: dict[str, int],
) -> str | tuple[int, tuple[int, int, int, int]]:
    if train_initializer in TRAIN_INITIALIZER_LINESTYLES:
        return TRAIN_INITIALIZER_LINESTYLES[train_initializer]
    fallback_linestyles: tuple[str, ...] = ("-.", ":")
    order_idx = initializer_order.get(train_initializer, 0)
    return fallback_linestyles[order_idx % len(fallback_linestyles)]


def _expand_ylim(
    ylim: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if ylim is None:
        return None
    low, high = ylim
    pad = 0.03 * (high - low)
    return (low - pad, high + pad)


def _get_series_entries(summary: pd.DataFrame) -> list[dict[str, object]]:
    has_multiple_eval_initializers = (
        len(summary["eval_initializer"].drop_duplicates()) > 1
    )
    method_order = {method: idx for idx, method in enumerate(METHOD_STYLES)}
    train_initializers = list(summary["train_initializer"].drop_duplicates())
    train_initializer_order = {
        initializer: idx
        for idx, initializer in enumerate(
            sorted(
                train_initializers,
                key=lambda initializer: (
                    0 if initializer in TRAIN_INITIALIZER_LABELS else 1,
                    TRAIN_INITIALIZER_LABELS.get(initializer, initializer),
                ),
            )
        )
    }
    eval_initializers = list(summary["eval_initializer"].drop_duplicates())
    eval_initializer_order = {
        initializer: idx for idx, initializer in enumerate(eval_initializers)
    }
    variant_order = {
        (
            str(train_initializer),
            str(eval_initializer),
        ): idx
        for idx, (train_initializer, eval_initializer) in enumerate(
            summary[["train_initializer", "eval_initializer"]]
            .drop_duplicates()
            .sort_values(
                ["train_initializer", "eval_initializer"],
                key=lambda col: col.map(
                    train_initializer_order
                    if col.name == "train_initializer"
                    else eval_initializer_order
                ).fillna(999),
            )
            .itertuples(index=False, name=None)
        )
    }
    unique_pairs = (
        summary[["method", "train_initializer", "eval_initializer"]]
        .drop_duplicates()
        .sort_values(
            ["method", "train_initializer", "eval_initializer"],
            key=lambda col: col.map(
                method_order
                if col.name == "method"
                else (
                    train_initializer_order
                    if col.name == "train_initializer"
                    else eval_initializer_order
                )
            ).fillna(999),
        )
    )
    entries: list[dict[str, object]] = []
    for row in unique_pairs.itertuples(index=False):
        method = str(row.method)
        train_initializer = str(row.train_initializer)
        eval_initializer = str(row.eval_initializer)
        base_style = _method_style(method).copy()
        base_style["linestyle"] = _train_initializer_linestyle(
            train_initializer,
            train_initializer_order,
        )
        variant_label = _train_initializer_label(train_initializer)
        if has_multiple_eval_initializers:
            variant_label = f"{variant_label}; eval={eval_initializer}"
        entries.append(
            {
                "method": method,
                "train_initializer": train_initializer,
                "eval_initializer": eval_initializer,
                "method_label": _method_label(method),
                "variant_label": variant_label,
                "variant_index": variant_order[
                    (train_initializer, eval_initializer)
                ],
                "n_variants": len(variant_order),
                "style": base_style,
            }
        )
    return entries


def _marker_x_positions(
    x: NDArray[np.floating],
    *,
    budget_mode: str,
    variant_index: int,
    n_variants: int,
) -> NDArray[np.floating]:
    if n_variants <= 1:
        return x

    centered_index = variant_index - (n_variants - 1) / 2
    if budget_mode == "soft":
        return x * np.exp(0.045 * centered_index)
    return x + 0.05 * centered_index


def _plot_metric_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    budget_mode: str,
    metric: str,
    title: str,
    series_entries: list[dict[str, object]],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    budget_column = _budget_column(budget_mode)
    x_label = _budget_label(budget_mode)
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    display_ylim = _expand_ylim(ylim)
    plotted_budgets: list[float] = []
    for entry in series_entries:
        group = _series_group(summary, entry, budget_column)
        if group.empty:
            continue
        plotted_budgets.extend(group[budget_column].to_list())
        _plot_series(
            ax,
            group,
            entry,
            mean_column=mean_column,
            std_column=std_column,
            display_ylim=display_ylim,
            budget_mode=budget_mode,
            budget_column=budget_column,
        )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    if display_ylim is not None:
        ax.set_ylim(*display_ylim)
    if budget_mode == "soft":
        ax.set_xscale("log")
    if plotted_budgets:
        _set_budget_axis_limits(ax, budget_mode, plotted_budgets)
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    ax.margins(x=0.02)


def _series_group(
    summary: pd.DataFrame,
    entry: dict[str, object],
    budget_column: str,
) -> pd.DataFrame:
    return summary[
        (summary["method"] == str(entry["method"]))
        & (summary["train_initializer"] == str(entry["train_initializer"]))
        & (summary["eval_initializer"] == str(entry["eval_initializer"]))
    ].sort_values(budget_column)


def _plot_series(
    ax: plt.Axes,
    group: pd.DataFrame,
    entry: dict[str, object],
    *,
    mean_column: str,
    std_column: str,
    display_ylim: tuple[float, float] | None,
    budget_mode: str,
    budget_column: str,
) -> None:
    style = entry["style"]
    x = group[budget_column].to_numpy()
    y = group[mean_column].to_numpy()
    yerr = group[std_column].to_numpy()
    marker_x = _marker_x_positions(
        x,
        budget_mode=budget_mode,
        variant_index=int(entry["variant_index"]),
        n_variants=int(entry["n_variants"]),
    )
    ax.plot(
        x,
        y,
        color=style["color"],
        linestyle=style["linestyle"],
    )
    ax.plot(
        marker_x,
        y,
        color=style["color"],
        linestyle="None",
        marker=style["marker"],
        markerfacecolor="white",
        markeredgewidth=1.6,
        zorder=3,
    )
    if group["runs"].max() <= 1:
        return
    lower = y - yerr
    upper = y + yerr
    if display_ylim is not None:
        lower = lower.clip(display_ylim[0], display_ylim[1])
        upper = upper.clip(display_ylim[0], display_ylim[1])
    ax.fill_between(
        x,
        lower,
        upper,
        color=style["color"],
        alpha=0.10,
        linewidth=0,
    )


def _set_budget_axis_limits(
    ax: plt.Axes,
    budget_mode: str,
    plotted_budgets: list[float],
) -> None:
    budgets = sorted(set(plotted_budgets))
    ax.set_xticks(budgets)
    if budget_mode == "soft":
        if len(budgets) == 1:
            ax.set_xlim(budgets[0] / 1.8, budgets[0] * 1.8)
        else:
            ax.set_xlim(min(budgets) * 0.8, max(budgets) * 1.25)
        return
    if len(budgets) == 1:
        ax.set_xlim(budgets[0] - 0.35, budgets[0] + 0.35)
    else:
        ax.set_xlim(min(budgets) - 0.1, max(budgets) + 0.1)


def _annotate_panels(axes: list[plt.Axes]) -> None:
    for idx, ax in enumerate(axes):
        ax.text(
            -0.14,
            1.08,
            f"{chr(ord('A') + idx)}",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )


def _finish_figure(
    fig: plt.Figure,
    series_entries: list[dict[str, object]],
    *,
    output_path: Path,
    formats: list[str],
    method_legend_ncol: int = 3,
) -> None:
    top_rect = 0.94
    methods_in_plot: dict[str, dict[str, object]] = {}
    variants_in_plot: dict[str, dict[str, object]] = {}
    for entry in series_entries:
        methods_in_plot.setdefault(str(entry["method"]), entry)
        variants_in_plot.setdefault(str(entry["variant_label"]), entry)

    method_handles = [
        Line2D(
            [],
            [],
            color=str(entry["style"]["color"]),
            linestyle="-",
            marker=str(entry["style"]["marker"]),
            markerfacecolor="white",
            markeredgewidth=1.6,
            label=str(entry["method_label"]),
        )
        for entry in methods_in_plot.values()
    ]
    if method_handles:
        method_legend_rows = (
            len(method_handles) + method_legend_ncol - 1
        ) // method_legend_ncol
        top_rect = max(0.78, 0.92 - 0.07 * max(0, method_legend_rows - 1))
        method_legend = fig.legend(
            handles=method_handles,
            labels=[handle.get_label() for handle in method_handles],
            title="Policy",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=min(method_legend_ncol, len(method_handles)),
        )
        fig.add_artist(method_legend)

    if len(variants_in_plot) > 1:
        variant_handles = [
            Line2D(
                [],
                [],
                color="#4C4C4C",
                linestyle=str(entry["style"]["linestyle"]),
                linewidth=2.2,
                label=str(entry["variant_label"]),
            )
            for entry in variants_in_plot.values()
        ]
        variant_legend = fig.legend(
            handles=variant_handles,
            labels=[handle.get_label() for handle in variant_handles],
            title="Training Data",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=min(len(variant_handles), 3),
            handlelength=3.0,
        )
        fig.add_artist(variant_legend)
        top_rect = min(top_rect, 0.72)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, top_rect))
    for fmt in formats:
        fig.savefig(
            output_path.with_suffix(f".{fmt}"),
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_accuracy(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
) -> None:
    series_entries = _get_series_entries(summary)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), squeeze=False)
    panels = [
        ("accuracy", "Overall Accuracy", (0.0, 1.0)),
        ("mean_cost", "Mean Acquisition Cost", None),
    ]
    ylabels = ["Accuracy", "Cost"]
    for ax, (metric, title, ylim), ylabel in zip(
        axes[0], panels, ylabels, strict=True
    ):
        _plot_metric_panel(
            ax,
            summary,
            budget_mode=budget_mode,
            metric=metric,
            title=title,
            series_entries=series_entries,
            ylabel=ylabel,
            ylim=ylim,
        )
    _annotate_panels(list(axes[0]))
    _finish_figure(
        fig,
        series_entries,
        output_path=output_path,
        formats=formats,
    )


def plot_behavior(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
) -> None:
    series_entries = _get_series_entries(summary)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1), squeeze=False)
    metrics = [
        ("first_action_context_rate", "First Action = Context", (0.0, 1.0)),
        ("rescue_episode_rate", "Episode Rescue Rate", (0.0, 1.0)),
    ]
    for ax, (metric, title, ylim) in zip(axes[0], metrics, strict=True):
        _plot_metric_panel(
            ax,
            summary,
            budget_mode=budget_mode,
            metric=metric,
            title=title,
            series_entries=series_entries,
            ylabel="Rate",
            ylim=ylim,
        )
    _annotate_panels(list(axes[0]))
    _finish_figure(
        fig,
        series_entries,
        output_path=output_path,
        formats=formats,
    )


def plot_risk(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
) -> None:
    series_entries = _get_series_entries(summary)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.9), squeeze=False)
    metrics = [
        ("risky_accuracy", "Risky-Context Accuracy", (0.0, 1.0)),
        ("risky_rescue_rate", "Risky Rescue Rate", (0.0, 1.0)),
        ("risky_unsafe_stop_rate", "Risky Unsafe-Stop Rate", (0.0, 1.0)),
        ("first_action_context_rate", "First Action = Context", (0.0, 1.0)),
    ]
    for ax, (metric, title, ylim) in zip(axes.flat, metrics, strict=True):
        _plot_metric_panel(
            ax,
            summary,
            budget_mode=budget_mode,
            metric=metric,
            title=title,
            series_entries=series_entries,
            ylabel="Rate",
            ylim=ylim,
        )
    _annotate_panels(list(axes.flat))
    _finish_figure(
        fig,
        series_entries,
        output_path=output_path,
        formats=formats,
        method_legend_ncol=2,
    )


def plot_paper_summary(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
) -> None:
    series_entries = _get_series_entries(summary)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5.9), squeeze=False)
    metrics = [
        ("accuracy", "Overall Accuracy", (0.0, 1.0)),
        ("risky_accuracy", "Risky-Context Accuracy", (0.0, 1.0)),
        ("risky_rescue_rate", "Risky Rescue Rate", (0.0, 1.0)),
        ("risky_unsafe_stop_rate", "Risky Unsafe-Stop Rate", (0.0, 1.0)),
    ]
    ylabels = ["Accuracy", "Accuracy", "Rate", "Rate"]
    for ax, (metric, title, ylim), ylabel in zip(
        axes.flat, metrics, ylabels, strict=True
    ):
        _plot_metric_panel(
            ax,
            summary,
            budget_mode=budget_mode,
            metric=metric,
            title=title,
            series_entries=series_entries,
            ylabel=ylabel,
            ylim=ylim,
        )
    _annotate_panels(list(axes.flat))
    _finish_figure(
        fig,
        series_entries,
        output_path=output_path,
        formats=formats,
        method_legend_ncol=2,
    )


def plot_shielding(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
) -> None:
    if not any(
        summary[f"{metric}_mean"].notna().any()
        for metric in SHIELD_METRIC_KEYS
    ):
        return

    series_entries = _get_series_entries(summary)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.1), squeeze=False)
    metrics = [
        ("proposed_stop_rate", "Proposed Stop Rate", (0.0, 1.0)),
        ("rejected_stop_rate", "Rejected Stop Rate", (0.0, 1.0)),
        ("forced_continue_rate", "Forced-Continue Rate", (0.0, 1.0)),
    ]
    for ax, (metric, title, ylim) in zip(axes[0], metrics, strict=True):
        _plot_metric_panel(
            ax,
            summary,
            budget_mode=budget_mode,
            metric=metric,
            title=title,
            series_entries=series_entries,
            ylabel="Rate",
            ylim=ylim,
        )
    _annotate_panels(list(axes[0]))
    _finish_figure(
        fig,
        series_entries,
        output_path=output_path,
        formats=formats,
        method_legend_ncol=2,
    )


def main() -> None:
    args = parse_args()
    apply_paper_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_initializers = (
        args.train_initializers
        if args.train_initializers is not None
        else (
            [args.train_initializer]
            if args.train_initializer is not None
            else ["cube_nm_ar"]
        )
    )

    per_run = collect_episode_rows(
        args.eval_results_root,
        budget_mode=args.budget_mode,
        train_initializers=train_initializers,
        eval_initializers=args.eval_initializers,
        dataset=args.dataset,
        methods=args.methods,
        rescue_action=2 + args.n_contexts * args.block_size,
    )
    if per_run.empty:
        msg = "No matching eval_data.csv files found."
        raise ValueError(msg)

    per_run.to_csv(args.output_dir / "per_run_summary.csv", index=False)
    aggregated = aggregate_runs(per_run, budget_mode=args.budget_mode)
    aggregated.to_csv(args.output_dir / "aggregate_summary.csv", index=False)

    plot_accuracy(
        aggregated,
        args.output_dir / "accuracy_by_budget",
        budget_mode=args.budget_mode,
        formats=args.formats,
    )
    plot_behavior(
        aggregated,
        args.output_dir / "behavior_by_budget",
        budget_mode=args.budget_mode,
        formats=args.formats,
    )
    plot_risk(
        aggregated,
        args.output_dir / "risk_by_budget",
        budget_mode=args.budget_mode,
        formats=args.formats,
    )
    plot_paper_summary(
        aggregated,
        args.output_dir / "paper_summary",
        budget_mode=args.budget_mode,
        formats=args.formats,
    )
    plot_shielding(
        aggregated,
        args.output_dir / "shield_by_budget",
        budget_mode=args.budget_mode,
        formats=args.formats,
    )


if __name__ == "__main__":
    main()
