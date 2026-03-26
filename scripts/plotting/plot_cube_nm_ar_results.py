# pyright: reportCallIssue=false, reportArgumentType=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportPrivateImportUsage=false
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotnine as p9

from afabench.eval.plotting_config import (
    COLOR_PALETTE_NAME,
    METHOD_NAME_MAPPING,
    PLOT_HEIGHT,
    PLOT_WIDTH,
)

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

TRAIN_INITIALIZER_LABELS = {
    "cold": "Full train support",
    "cube_nm_ar": "Rescue-censored train",
    "cube_nm_ar_mar": "MAR control",
    "cube_nm_ar_p01": r"$p_\mathrm{miss}=0.1$",
    "cube_nm_ar_p05": r"$p_\mathrm{miss}=0.5$",
    "cube_nm_ar_p07": r"$p_\mathrm{miss}=0.7$",
    "mcar_p03": "MCAR 30%",
    "mar_p03": "MAR 30%",
    "mnar_logistic_p03": "MNAR logistic 30%",
}

# Mapping from train_initializer to (p_miss, mechanism).
# cube_nm_ar defaults to MNAR at p=0.3.
INITIALIZER_TO_PMISS: dict[str, tuple[float, str]] = {
    "cold": (0.0, "MNAR"),
    "cube_nm_ar": (0.3, "MNAR"),
    "cube_nm_ar_p01": (0.1, "MNAR"),
    "cube_nm_ar_p05": (0.5, "MNAR"),
    "cube_nm_ar_p07": (0.7, "MNAR"),
    "cube_nm_ar_mar": (0.3, "MAR"),
}

# Representative soft budgets per method family.
REPRESENTATIVE_SOFT_BUDGETS: dict[str, float] = {
    "aaco_full": 1.2,
    "aaco_zero_fill": 1.2,
    "aaco_mask_aware": 1.2,
    "aaco_dr": 1.2,
    "gadgil2023": 0.01,
    "gadgil2023_ipw_feature_marginal": 0.01,
    "ol_with_mask": 0.01,
}

METRIC_DISPLAY = {
    "accuracy": "Overall Accuracy",
    "risky_accuracy": "Risky-Context Accuracy",
    "risky_unsafe_stop_rate": "Risky Unsafe-Stop Rate",
    "mean_cost": "Mean Acquisition Cost",
    "risky_rescue_rate": "Risky Rescue Rate",
    "first_action_context_rate": "First Action = Context",
    "rescue_episode_rate": "Episode Rescue Rate",
    "forced_stop_rate": "Forced-Stop Rate",
}

PUBLICATION_METHOD_SPECS: dict[str, dict[str, str | None]] = {
    "gadgil2023": {
        "baseline_label": "DIME baseline",
        "curve_label": None,
    },
    "gadgil2023_ipw_feature_marginal": {
        "baseline_label": None,
        "curve_label": "DIME + IPW",
    },
    "aaco_full": {
        "baseline_label": "AACO baseline",
        "curve_label": None,
    },
    "aaco_zero_fill": {
        "baseline_label": None,
        "curve_label": "AACO + zero-fill",
    },
    "aaco_mask_aware": {
        "baseline_label": None,
        "curve_label": "AACO + mask-aware",
    },
    "aaco_dr": {
        "baseline_label": None,
        "curve_label": "AACO + DR",
    },
    "ol_with_mask": {
        "baseline_label": "OL-MFRL baseline",
        "curve_label": None,
    },
}

CURVE_LABEL_ORDER = [
    "DIME + IPW",
    "AACO + zero-fill",
    "AACO + mask-aware",
    "AACO + DR",
]

BASELINE_LABEL_ORDER = [
    "DIME baseline",
    "AACO baseline",
    "OL-MFRL baseline",
]

BASELINE_LINETYPES = {
    "DIME baseline": "solid",
    "AACO baseline": "dashed",
    "OL-MFRL baseline": "dashdot",
}


# ── Data loading (kept from original) ────────────────────────


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
            "aaco_dr",
            "ol_with_mask",
        ],
    )
    parser.add_argument("--n-contexts", type=int, default=5)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
    )
    parser.add_argument(
        "--from-aggregate",
        type=Path,
        default=None,
        help=(
            "Load from a pre-computed aggregate_summary.csv "
            "instead of scanning the filesystem."
        ),
    )
    parser.add_argument(
        "--shield-aggregate",
        type=Path,
        default=None,
        help=(
            "Path to aggregate_summary.csv for the shielded "
            "condition (enables shield comparison plot)."
        ),
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


def _publication_method_spec(
    method: str,
) -> dict[str, str | None]:
    default_label = METHOD_NAME_MAPPING.get(method, method)
    return PUBLICATION_METHOD_SPECS.get(
        method,
        {
            "baseline_label": None,
            "curve_label": default_label,
        },
    )


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
        r"train_seed-(\d+)\+train_hard_budget-([^+/]+)"
        r"\+train_soft_budget_param-([^/]+)/"
        r"eval_seed-(\d+)\+eval_hard_budget-([^+/]+)"
        r"\+eval_soft_budget_param-([^/]+)/"
        r"eval_data\.csv$"
    )

    for train_initializer in train_initializers:
        for eval_initializer in eval_initializers:
            relative_init_root = Path(
                "train_initializer-"
                f"{train_initializer}"
                f"+eval_initializer-{eval_initializer}"
            )
            init_root = (
                root
                if root.name == relative_init_root.name
                else root / relative_init_root
            )
            for method in methods:
                glob_pattern = (
                    f"{method}/dataset-{dataset}"
                    f"+instance_idx-*/**/eval_data.csv"
                )
                for path in init_root.glob(glob_pattern):
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
    budget_col = _budget_column(budget_mode)
    aggregated = (
        summary.groupby(
            [
                "train_initializer",
                "eval_initializer",
                "method",
                budget_col,
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
            risky_episode_rate_mean=(
                "risky_episode_rate",
                "mean",
            ),
            risky_episode_rate_std=("risky_episode_rate", "std"),
            risky_accuracy_mean=("risky_accuracy", "mean"),
            risky_accuracy_std=("risky_accuracy", "std"),
            risky_rescue_rate_mean=("risky_rescue_rate", "mean"),
            risky_rescue_rate_std=("risky_rescue_rate", "std"),
            risky_unsafe_stop_rate_mean=(
                "risky_unsafe_stop_rate",
                "mean",
            ),
            risky_unsafe_stop_rate_std=(
                "risky_unsafe_stop_rate",
                "std",
            ),
            context_episode_rate_mean=(
                "context_episode_rate",
                "mean",
            ),
            context_episode_rate_std=(
                "context_episode_rate",
                "std",
            ),
            rescue_episode_rate_mean=(
                "rescue_episode_rate",
                "mean",
            ),
            rescue_episode_rate_std=("rescue_episode_rate", "std"),
            first_action_context_rate_mean=(
                "first_action_context_rate",
                "mean",
            ),
            first_action_context_rate_std=(
                "first_action_context_rate",
                "std",
            ),
            proposed_stop_rate_mean=("proposed_stop_rate", "mean"),
            proposed_stop_rate_std=("proposed_stop_rate", "std"),
            rejected_stop_rate_mean=("rejected_stop_rate", "mean"),
            rejected_stop_rate_std=("rejected_stop_rate", "std"),
            forced_continue_rate_mean=(
                "forced_continue_rate",
                "mean",
            ),
            forced_continue_rate_std=(
                "forced_continue_rate",
                "std",
            ),
        )
        .sort_values(
            [
                "method",
                "train_initializer",
                "eval_initializer",
                budget_col,
            ]
        )
    )
    std_columns = [c for c in aggregated.columns if c.endswith("_std")]
    aggregated[std_columns] = aggregated[std_columns].fillna(0.0)
    return aggregated


# ── Plotting helpers (plotnine) ──────────────────────────────


def _melt_to_long(
    agg: pd.DataFrame,
    metrics: list[str],
    budget_col: str,
) -> pd.DataFrame:
    """Melt aggregated data into long form for faceted plotting."""
    frames = []
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in agg.columns:
            continue
        sub = agg[
            [
                "train_initializer",
                "eval_initializer",
                "method",
                budget_col,
                "runs",
                mean_col,
                std_col,
            ]
        ].copy()
        sub = sub.rename(
            columns={
                mean_col: "mean_metric",
                std_col: "std_metric",
            }
        )
        sub["metric_name"] = METRIC_DISPLAY.get(metric, metric)
        sem = sub["std_metric"] / np.sqrt(sub["runs"].clip(lower=1))
        sub["low_metric"] = sub["mean_metric"] - sem
        sub["high_metric"] = sub["mean_metric"] + sem
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def _apply_method_labels(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data = data.copy()
    data["method"] = data["method"].map(
        lambda m: METHOD_NAME_MAPPING.get(m, m)
    )
    return data


def _method_breaks(df: pd.DataFrame) -> list[str]:
    order = list(METHOD_NAME_MAPPING.values())
    available = df["method"].unique().tolist()
    return [m for m in order if m in available]


def _add_publication_method_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    data = data.copy()
    data["baseline_label"] = data["method"].map(
        lambda method: _publication_method_spec(method)["baseline_label"]
    )
    data["curve_label"] = data["method"].map(
        lambda method: _publication_method_spec(method)["curve_label"]
    )
    return data


def _make_budget_sweep_plot(
    long_df: pd.DataFrame,
    *,
    budget_col: str,
    budget_label: str,
    n_metrics: int,
) -> p9.ggplot:
    long_df = _apply_method_labels(long_df)
    breaks = _method_breaks(long_df)

    n_rows = (n_metrics + 1) // 2
    fig_height = max(PLOT_HEIGHT, 2.8 * n_rows)

    plot = (
        p9.ggplot(
            long_df,
            p9.aes(
                x=budget_col,
                y="mean_metric",
                color="method",
                fill="method",
            ),
        )
        + p9.geom_line()
        + p9.geom_point(size=2.5)
        + p9.geom_ribbon(
            p9.aes(ymin="low_metric", ymax="high_metric"),
            alpha=0.1,
            size=0.0,
        )
        + p9.facet_wrap("metric_name", scales="free_y", ncol=2)
        + p9.scale_color_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=breaks,
        )
        + p9.scale_fill_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=breaks,
        )
        + p9.labs(
            x=budget_label,
            y="Value",
            color="Policy",
            fill="Policy",
        )
        + p9.theme(figure_size=(PLOT_WIDTH * 0.75, fig_height))
    )
    return plot


def _save_plot(
    plot: p9.ggplot,
    path: Path,
    *,
    width: float,
    height: float,
    formats: list[str],
) -> None:
    for fmt in formats:
        plot.save(
            path.with_suffix(f".{fmt}"),
            width=width,
            height=height,
        )


def _filter_representative_budgets(
    agg: pd.DataFrame,
    budget_col: str,
    *,
    budget_mode: str,
) -> pd.DataFrame:
    if budget_mode == "hard":
        representative = agg.groupby("method")[budget_col].transform("max")
        return pd.DataFrame(agg[agg[budget_col] == representative]).copy()

    # Keep only rows at each method's representative soft budget.
    mask = agg.apply(
        lambda row: (
            row["method"] in REPRESENTATIVE_SOFT_BUDGETS
            and row[budget_col] == REPRESENTATIVE_SOFT_BUDGETS[row["method"]]
        ),
        axis=1,
    )
    return pd.DataFrame(agg[mask]).copy()


def _melt_degradation(
    agg: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    # Add p_miss and mechanism columns from train_initializer.
    agg = agg.copy()
    agg["p_miss"] = agg["train_initializer"].map(
        lambda ti: INITIALIZER_TO_PMISS.get(ti, (None, None))[0]
    )
    agg["mechanism"] = agg["train_initializer"].map(
        lambda ti: INITIALIZER_TO_PMISS.get(ti, (None, None))[1]
    )
    agg = agg.dropna(subset=["p_miss"])

    frames = []
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col not in agg.columns:
            continue
        cols = [
            "method",
            "train_initializer",
            "p_miss",
            "mechanism",
            mean_col,
            std_col,
        ]
        if "runs" in agg.columns:
            cols.append("runs")
        sub = agg[cols].copy()
        sub = sub.rename(
            columns={
                mean_col: "mean_metric",
                std_col: "std_metric",
            }
        )
        sub["metric_name"] = METRIC_DISPLAY.get(metric, metric)
        n_runs = sub["runs"].clip(lower=1) if "runs" in sub.columns else 1
        sem = sub["std_metric"] / np.sqrt(n_runs)
        sub["low_metric"] = sub["mean_metric"] - sem
        sub["high_metric"] = sub["mean_metric"] + sem
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _make_degradation_plot(
    long_df: pd.DataFrame,
    *,
    n_metrics: int,
) -> p9.ggplot:
    publication_df = _add_publication_method_columns(long_df)
    baseline = publication_df[
        (publication_df["train_initializer"] == "cold")
        & publication_df["baseline_label"].notna()
    ].copy()
    mnar = publication_df[
        (publication_df["train_initializer"] != "cold")
        & (publication_df["mechanism"] == "MNAR")
        & publication_df["curve_label"].notna()
    ].copy()
    mar = publication_df[
        (publication_df["train_initializer"] != "cold")
        & (publication_df["mechanism"] == "MAR")
        & publication_df["curve_label"].notna()
    ].copy()
    curve_breaks = [
        label
        for label in CURVE_LABEL_ORDER
        if label in mnar["curve_label"].unique().tolist()
    ]
    baseline_breaks = [
        label
        for label in BASELINE_LABEL_ORDER
        if label in baseline["baseline_label"].unique().tolist()
    ]

    n_rows = (n_metrics + 1) // 2
    fig_height = max(PLOT_HEIGHT, 2.8 * n_rows)

    plot = (
        p9.ggplot(
            mnar,
            p9.aes(
                x="p_miss",
                y="mean_metric",
                color="curve_label",
                fill="curve_label",
            ),
        )
        + p9.geom_line()
        + p9.geom_point(size=2.5)
        + p9.geom_ribbon(
            p9.aes(ymin="low_metric", ymax="high_metric"),
            alpha=0.1,
            size=0.0,
            show_legend=False,
        )
        + p9.facet_wrap("metric_name", scales="free_y", ncol=2)
        + p9.scale_color_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=curve_breaks,
        )
        + p9.scale_fill_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=curve_breaks,
        )
        + p9.scale_x_continuous(
            labels=lambda xs: [f"{x:.0%}" for x in xs],
        )
        + p9.labs(
            x="Training missingness rate (MNAR)",
            y="Value",
            color="Mitigations",
            linetype="Baselines",
        )
        + p9.theme(figure_size=(PLOT_WIDTH * 0.75, fig_height))
    )
    if not baseline.empty:
        plot += p9.geom_hline(
            data=baseline,
            mapping=p9.aes(
                yintercept="mean_metric",
                linetype="baseline_label",
            ),
            color="black",
            alpha=0.75,
            size=0.65,
        )
        plot += p9.scale_linetype_manual(
            values=BASELINE_LINETYPES,
            breaks=baseline_breaks,
        )
    if not mar.empty:
        plot += p9.geom_point(
            data=mar,
            mapping=p9.aes(
                x="p_miss",
                y="mean_metric",
                color="curve_label",
            ),
            shape="^",
            size=4,
            stroke=0.8,
            fill="white",
        )
    return plot


def plot_missingness_degradation(
    aggregated: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
    metrics: list[str] | None = None,
) -> None:
    if budget_mode != "soft":
        print(
            "Skipping missingness degradation plot: "
            "CUBE-NM-AR publication comparison is soft-budget only."
        )
        return

    if metrics is None:
        metrics = [
            "accuracy",
            "risky_accuracy",
            "risky_unsafe_stop_rate",
            "mean_cost",
        ]
    budget_col = _budget_column(budget_mode)
    filtered = _filter_representative_budgets(
        aggregated,
        budget_col,
        budget_mode=budget_mode,
    )
    if filtered.empty:
        print("No rows at representative budgets; skipping degradation plot.")
        return

    long = _melt_degradation(filtered, metrics)
    if long.empty:
        return

    n_metrics = long["metric_name"].nunique()
    n_rows = (n_metrics + 1) // 2
    fig_height = max(PLOT_HEIGHT, 2.8 * n_rows)
    fig_width = PLOT_WIDTH * 0.75

    plot = _make_degradation_plot(long, n_metrics=n_metrics)
    _save_plot(
        plot,
        output_path,
        width=fig_width,
        height=fig_height,
        formats=formats,
    )


def plot_shield_comparison(
    unshielded: pd.DataFrame,
    shielded: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
    metrics: list[str] | None = None,
) -> None:
    # Figure C: side-by-side unshielded vs shielded.
    if metrics is None:
        metrics = [
            "accuracy",
            "risky_accuracy",
            "risky_unsafe_stop_rate",
            "mean_cost",
        ]
    budget_col = _budget_column(budget_mode)

    unshielded_f = _filter_representative_budgets(
        unshielded,
        budget_col,
        budget_mode=budget_mode,
    )
    shielded_f = _filter_representative_budgets(
        shielded,
        budget_col,
        budget_mode=budget_mode,
    )

    unshielded_long = _melt_degradation(unshielded_f, metrics)
    shielded_long = _melt_degradation(shielded_f, metrics)

    if unshielded_long.empty and shielded_long.empty:
        print("No data for shield comparison; skipping.")
        return

    unshielded_long["shield_status"] = "Unshielded"
    shielded_long["shield_status"] = "Shielded"
    combined = pd.concat([unshielded_long, shielded_long], ignore_index=True)
    combined = _add_publication_method_columns(combined)
    combined = combined[
        (combined["mechanism"] == "MNAR") & combined["curve_label"].notna()
    ].copy()
    if combined.empty:
        return

    breaks = [
        label
        for label in CURVE_LABEL_ORDER
        if label in combined["curve_label"].unique().tolist()
    ]

    n_metrics = combined["metric_name"].nunique()
    fig_height = max(PLOT_HEIGHT, 2.5 * n_metrics)
    fig_width = PLOT_WIDTH * 0.75

    plot = (
        p9.ggplot(
            combined,
            p9.aes(
                x="p_miss",
                y="mean_metric",
                color="curve_label",
                fill="curve_label",
            ),
        )
        + p9.geom_line()
        + p9.geom_point(size=2.5)
        + p9.geom_ribbon(
            p9.aes(ymin="low_metric", ymax="high_metric"),
            alpha=0.1,
            size=0.0,
            show_legend=False,
        )
        + p9.facet_grid(
            "metric_name ~ shield_status",
            scales="free_y",
        )
        + p9.scale_color_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=breaks,
        )
        + p9.scale_fill_brewer(
            type="qual",
            palette=COLOR_PALETTE_NAME,
            breaks=breaks,
        )
        + p9.scale_x_continuous(
            labels=lambda xs: [f"{x:.0%}" for x in xs],
        )
        + p9.labs(
            x="Training missingness rate (MNAR)",
            y="Value",
            color="Mitigations",
        )
        + p9.theme(figure_size=(fig_width, fig_height))
    )
    _save_plot(
        plot,
        output_path,
        width=fig_width,
        height=fig_height,
        formats=formats,
    )


# ── High-level plot functions ────────────────────────────────


def plot_budget_sweep(
    aggregated: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
    metrics: list[str] | None = None,
) -> None:
    if metrics is None:
        metrics = [
            "accuracy",
            "risky_accuracy",
            "risky_unsafe_stop_rate",
            "mean_cost",
        ]
    budget_col = _budget_column(budget_mode)
    budget_label = _budget_label(budget_mode)

    long = _melt_to_long(aggregated, metrics, budget_col)
    if long.empty:
        return

    n_metrics = long["metric_name"].nunique()
    n_rows = (n_metrics + 1) // 2
    fig_height = max(PLOT_HEIGHT, 2.8 * n_rows)
    fig_width = PLOT_WIDTH * 0.75

    plot = _make_budget_sweep_plot(
        long,
        budget_col=budget_col,
        budget_label=budget_label,
        n_metrics=n_metrics,
    )
    _save_plot(
        plot,
        output_path,
        width=fig_width,
        height=fig_height,
        formats=formats,
    )


def plot_accuracy_and_cost(
    aggregated: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
) -> None:
    plot_budget_sweep(
        aggregated,
        output_path,
        budget_mode=budget_mode,
        formats=formats,
        metrics=["accuracy", "mean_cost"],
    )


def plot_paper_summary(
    aggregated: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
) -> None:
    plot_budget_sweep(
        aggregated,
        output_path,
        budget_mode=budget_mode,
        formats=formats,
        metrics=[
            "accuracy",
            "risky_accuracy",
            "risky_rescue_rate",
            "risky_unsafe_stop_rate",
        ],
    )


def plot_behavior(
    aggregated: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
) -> None:
    plot_budget_sweep(
        aggregated,
        output_path,
        budget_mode=budget_mode,
        formats=formats,
        metrics=[
            "first_action_context_rate",
            "rescue_episode_rate",
        ],
    )


def plot_risk(
    aggregated: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
) -> None:
    plot_budget_sweep(
        aggregated,
        output_path,
        budget_mode=budget_mode,
        formats=formats,
        metrics=[
            "risky_accuracy",
            "risky_rescue_rate",
            "risky_unsafe_stop_rate",
            "first_action_context_rate",
        ],
    )


def plot_shielding(
    aggregated: pd.DataFrame,
    output_path: Path,
    *,
    budget_mode: str,
    formats: list[str],
) -> None:
    has_data = any(
        aggregated[f"{m}_mean"].notna().any() for m in SHIELD_METRIC_KEYS
    )
    if not has_data:
        return
    plot_budget_sweep(
        aggregated,
        output_path,
        budget_mode=budget_mode,
        formats=formats,
        metrics=list(SHIELD_METRIC_KEYS),
    )


# ── Main ─────────────────────────────────────────────────────


def _load_or_collect(
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.from_aggregate is not None:
        return pd.read_csv(args.from_aggregate)

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

    per_run.to_csv(
        args.output_dir / "per_run_summary.csv",
        index=False,
    )
    aggregated = aggregate_runs(per_run, budget_mode=args.budget_mode)
    aggregated.to_csv(
        args.output_dir / "aggregate_summary.csv",
        index=False,
    )
    return aggregated


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregated = _load_or_collect(args)

    multi_init = len(aggregated["train_initializer"].unique()) > 1

    # Budget sweep plots only make sense for a single
    # train_initializer; skip when comparison data is loaded.
    if not multi_init:
        plot_accuracy_and_cost(
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
    else:
        print(
            "Multiple train_initializers detected; "
            "skipping per-initializer budget sweep plots."
        )

    if multi_init:
        plot_missingness_degradation(
            aggregated,
            args.output_dir / "missingness_degradation",
            budget_mode=args.budget_mode,
            formats=args.formats,
        )
    else:
        print(
            "Single train_initializer detected; "
            "skipping missingness degradation plot."
        )

    # Shield comparison (Figure C).
    if args.shield_aggregate is not None:
        shielded = pd.read_csv(args.shield_aggregate)
        plot_shield_comparison(
            aggregated,
            shielded,
            args.output_dir / "shield_comparison",
            budget_mode=args.budget_mode,
            formats=args.formats,
        )


if __name__ == "__main__":
    main()
