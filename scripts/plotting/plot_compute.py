"""Plot paired restoration gain against generative/direct wall-time ratio."""

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
    "dime": "#1baf7a",
    "ol_with_mask": "#eb6834",
    "ol_full_state": "#8c5fd3",
}
METHOD_LABELS = {
    "aaco": "AACO",
    "dime": "DIME",
    "ol_with_mask": "OL + mask",
    "ol_full_state": "OL + full state",
}
DATASET_MARKERS = {
    "cube": "o",
    "cube_nm": "s",
    "cube_nonuniform_costs": "D",
    "heart_disease": "P",
    "actg": "^",
    "diabetes": "v",
    "nhanes_mortality": "X",
}
DATASET_LABELS = {
    "cube": "CUBE",
    "cube_nm": "CUBE-NM",
    "cube_nonuniform_costs": "CUBE-NUC",
    "heart_disease": "Heart disease",
    "actg": "ACTG175",
    "diabetes": "Diabetes",
    "nhanes_mortality": "NHANES mortality",
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
        "architecture",
        "torch",
        "torch_cuda",
    ]
    training = _rows(compute, _column(compute, "rule") == "train_method")
    training = _rows(training, _column(training, "method").isin(METHOD_COLORS))
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
    figure, axis = plt.subplots(figsize=(7.1, 3.4))
    for dataset, marker in DATASET_MARKERS.items():
        for method, color in METHOD_COLORS.items():
            subset = _rows(
                frame,
                (_column(frame, "dataset") == dataset)
                & (_column(frame, "method") == method),
            )
            if subset.empty:
                continue
            axis.scatter(
                subset["wall_time_ratio"],
                subset["restoration_gain"],
                marker=marker,
                s=22,
                color=color,
                edgecolor=SURFACE,
                linewidth=0.4,
                alpha=0.55,
            )
    axis.axvline(1.0, color=GRID, linewidth=0.9)
    axis.axhline(0.0, color=GRID, linewidth=0.9)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Generative / direct wall time (paired, fixed budget)")
    axis.set_ylabel("Restoration gain in the dataset's primary metric")
    axis.grid(True, color=GRID, linewidth=0.4, alpha=0.55)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    method_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=color,
            label=METHOD_LABELS[method],
        )
        for method, color in METHOD_COLORS.items()
    ]
    dataset_handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            color=INK_MUTED,
            label=DATASET_LABELS[dataset],
        )
        for dataset, marker in DATASET_MARKERS.items()
        if dataset in set(_column(frame, "dataset"))
    ]
    figure.legend(
        handles=[*method_handles, *dataset_handles],
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=7,
        bbox_to_anchor=(0.5, 0.0),
    )
    figure.subplots_adjust(left=0.11, right=0.99, top=0.98, bottom=0.29)
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
