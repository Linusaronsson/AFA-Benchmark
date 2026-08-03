"""Collect every missing-data result into one auditable table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Collection

KEY_COLUMNS = [
    "split",
    "dataset",
    "method",
    "mechanism",
    "p",
    "strategy",
    "instance",
    "train_hard_budget",
    "eval_hard_budget",
]
COVERAGE_COLUMNS = [
    "split",
    "dataset",
    "method",
    "mechanism",
    "p",
    "strategy",
    "train_hard_budget",
    "eval_hard_budget",
]
ACCURACY_DATASETS = {
    "cube",
    "cube_nm",
    "cube_nonuniform_costs",
}


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast("pd.Series", frame[name])


def _source_frames(
    summary_root: Path,
    namespaces: Collection[str] | None = None,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    selected = set(namespaces) if namespaces is not None else None
    for path in sorted(summary_root.glob("*/*/instance_metrics.csv")):
        namespace = path.parent.name
        if selected is not None and namespace not in selected:
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["split"] = path.parents[1].name
        frame["namespace"] = namespace
        frame["source_file"] = str(path)
        frames.append(frame)
    return frames


def build_inventory(
    summary_root: Path,
    namespaces: Collection[str] | None = None,
) -> pd.DataFrame:
    """Return all result rows without silently resolving duplicate cells."""
    frames = _source_frames(summary_root, namespaces)
    if not frames:
        qualifier = (
            f" for namespaces {sorted(namespaces)}"
            if namespaces is not None
            else ""
        )
        msg = (
            f"No instance_metrics.csv files found below {summary_root}"
            f"{qualifier}."
        )
        raise FileNotFoundError(msg)
    inventory = pd.concat(frames, ignore_index=True, sort=False)

    missing = sorted(set(KEY_COLUMNS) - set(inventory.columns))
    if missing:
        msg = "Missing result key columns: " + ", ".join(missing)
        raise ValueError(msg)

    strategy = _column(inventory, "strategy").astype(str)
    inventory["training_condition"] = strategy.replace(
        {"restricted": "direct"}
    )
    mechanism = _column(inventory, "mechanism").astype(str)
    inventory["missingness_provenance"] = np.select(
        [mechanism == "none", mechanism == "native"],
        ["complete", "native"],
        default="induced",
    )

    accuracy = _column(inventory, "dataset").isin(ACCURACY_DATASETS)
    inventory["primary_metric"] = np.where(
        accuracy,
        "accuracy",
        "f_score",
    )
    inventory["primary_score"] = np.where(
        accuracy,
        _column(inventory, "accuracy"),
        _column(inventory, "f_score"),
    )
    primary = cast(
        "pd.Series",
        pd.to_numeric(
            _column(inventory, "primary_score"),
            errors="coerce",
        ),
    )
    samples = cast(
        "pd.Series",
        pd.to_numeric(
            _column(inventory, "n_samples"),
            errors="coerce",
        ),
    )
    inventory["finite_primary_score"] = np.isfinite(primary)
    inventory["positive_sample_count"] = samples > 0
    inventory["duplicate_sources"] = inventory.groupby(
        KEY_COLUMNS,
        dropna=False,
    )["namespace"].transform("nunique")

    identity = inventory[KEY_COLUMNS].astype(str).agg("|".join, axis=1)
    inventory["result_key"] = identity
    inventory["ready_for_analysis"] = (
        _column(inventory, "finite_primary_score")
        & _column(inventory, "positive_sample_count")
        & (_column(inventory, "duplicate_sources") == 1)
    )
    return inventory.sort_values(
        [*KEY_COLUMNS, "namespace"],
        kind="stable",
    ).reset_index(drop=True)


def build_coverage(inventory: pd.DataFrame) -> pd.DataFrame:
    """Summarize instances and sources available for each scientific cell."""
    grouped = inventory.groupby(COVERAGE_COLUMNS, dropna=False)
    coverage = grouped.agg(
        n_rows=("result_key", "size"),
        n_instances=("instance", "nunique"),
        n_sources=("namespace", "nunique"),
        all_scores_finite=("finite_primary_score", "all"),
        all_sample_counts_positive=("positive_sample_count", "all"),
    ).reset_index()
    sources = grouped["namespace"].agg(
        lambda values: ",".join(sorted(set(map(str, values))))
    )
    instances = grouped["instance"].agg(
        lambda values: ",".join(
            map(str, sorted(set(map(int, values)))),
        )
    )
    coverage["sources"] = sources.to_numpy()
    coverage["instances"] = instances.to_numpy()
    coverage["complete_five_instances"] = (
        (_column(coverage, "n_instances") == 5)
        & _column(coverage, "all_scores_finite")
        & _column(coverage, "all_sample_counts_positive")
    )
    return coverage.sort_values(COVERAGE_COLUMNS, kind="stable").reset_index(
        drop=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=Path("extra/output/missing_data/summary"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extra/output/missing_data/analysis/inventory"),
    )
    parser.add_argument(
        "--namespace",
        action="append",
        dest="namespaces",
        help="Include this exact namespace; repeat for multiple namespaces.",
    )
    arguments = parser.parse_args()

    inventory = build_inventory(arguments.summary_root, arguments.namespaces)
    coverage = build_coverage(inventory)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_parquet(
        arguments.output_dir / "results.parquet",
        index=False,
    )
    inventory.to_csv(arguments.output_dir / "results.csv", index=False)
    coverage.to_csv(arguments.output_dir / "coverage.csv", index=False)

    duplicates = int((_column(inventory, "duplicate_sources") > 1).sum())
    print(f"rows: {len(inventory)}")
    print(f"scientific cells: {len(coverage)}")
    print(f"rows with duplicate sources: {duplicates}")
    print(f"wrote {arguments.output_dir}")


if __name__ == "__main__":
    main()
