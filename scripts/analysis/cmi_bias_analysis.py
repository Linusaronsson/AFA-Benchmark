# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportMissingTypeArgument=false
"""
Analyze per-feature CMI bias from DIME under missing training data.

Loads cmi_log.pkl files produced by the instrumented DIME evaluation,
compares CMI estimates between full-data and missing-data training, and
generates scatter/bar plots of CMI deviation.

Usage:
    python scripts/analysis/cmi_bias_analysis.py \
        --results-dir extra/output/eval_results \
        --eval-split val \
        --output-dir extra/output/analysis/cmi_bias
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

INITIALIZER_META = {
    "cold": ("none", 0.0),
    "mcar_p01": ("MCAR", 0.1),
    "mcar_p03": ("MCAR", 0.3),
    "mcar_p05": ("MCAR", 0.5),
    "mar_p01": ("MAR", 0.1),
    "mar_p03": ("MAR", 0.3),
    "mar_p05": ("MAR", 0.5),
    "mnar_logistic_p01": ("MNAR", 0.1),
    "mnar_logistic_p03": ("MNAR", 0.3),
    "mnar_logistic_p05": ("MNAR", 0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("extra/output/eval_results"),
    )
    parser.add_argument("--eval-split", default="val")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extra/output/analysis/cmi_bias"),
    )
    return parser.parse_args()


def find_cmi_logs(results_dir: Path, eval_split: str) -> list[dict]:
    """Find all cmi_log.pkl files and extract metadata from paths."""
    records = []
    pattern = f"eval_split-{eval_split}/**/gadgil2023/**/cmi_log.pkl"
    for pkl_path in sorted(results_dir.glob(pattern)):
        path_str = str(pkl_path)
        # Extract train initializer from the path.
        init_match = re.search(r"train_init-([^+/]+)", path_str)
        train_init = init_match.group(1) if init_match else "unknown"
        # Extract dataset.
        ds_match = re.search(r"dataset-([^+/]+)", path_str)
        dataset = ds_match.group(1) if ds_match else "unknown"
        # Extract instance seed.
        seed_match = re.search(r"train_seed-(\d+)", path_str)
        train_seed = int(seed_match.group(1)) if seed_match else 0

        records.append(
            {
                "path": pkl_path,
                "train_initializer": train_init,
                "dataset": dataset,
                "train_seed": train_seed,
            }
        )
    return records


def load_cmi_log(path: Path) -> list[dict]:
    """Load a cmi_log.pkl file."""
    with path.open("rb") as f:
        return pickle.load(f)  # noqa: S301


def compute_feature_cmi_stats(
    cmi_log: list[dict],
) -> np.ndarray:
    """
    Compute mean per-feature CMI from a log.

    Returns array of shape (n_features,) with the mean CMI score for each feature.
    """
    all_cmi = []
    for entry in cmi_log:
        pred_cmi = entry["pred_cmi"].numpy()  # (batch, n_features)
        feature_mask = entry["feature_mask"].numpy()  # (batch, n_features)
        # Mask out already-acquired features (they have large negative scores).
        masked_cmi = np.where(feature_mask > 0.5, np.nan, pred_cmi)
        all_cmi.append(masked_cmi)
    stacked = np.concatenate(all_cmi, axis=0)  # (total_steps, n_features)
    return np.nanmean(stacked, axis=0)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Searching for CMI log files...")
    records = find_cmi_logs(args.results_dir, args.eval_split)
    if not records:
        print(
            "No cmi_log.pkl files found. Make sure to run evaluation with log_cmi=true."
        )
        return
    print(f"Found {len(records)} CMI log files.")

    rows = []
    for rec in records:
        cmi_log = load_cmi_log(rec["path"])
        if not cmi_log:
            continue
        mean_cmi = compute_feature_cmi_stats(cmi_log)
        n_features = len(mean_cmi)
        mechanism, rate = INITIALIZER_META.get(
            rec["train_initializer"], ("unknown", -1)
        )
        rows.extend(
            {
                "dataset": rec["dataset"],
                "train_initializer": rec["train_initializer"],
                "mechanism": mechanism,
                "miss_rate": rate,
                "train_seed": rec["train_seed"],
                "feature_idx": feat_idx,
                "mean_cmi": mean_cmi[feat_idx],
            }
            for feat_idx in range(n_features)
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No CMI data extracted.")
        return

    # Compute per-feature CMI under baseline.
    baseline = df[df["train_initializer"] == "cold"].copy()
    baseline_cmi = (
        baseline.groupby(["dataset", "feature_idx"])["mean_cmi"]
        .mean()
        .reset_index()
        .rename(columns={"mean_cmi": "baseline_cmi"})
    )

    # Merge and compute deviation.
    merged = df.merge(baseline_cmi, on=["dataset", "feature_idx"], how="left")
    merged["cmi_deviation"] = merged["mean_cmi"] - merged["baseline_cmi"]

    # Save full per-feature data.
    detail_path = args.output_dir / "cmi_per_feature_detail.csv"
    merged.to_csv(detail_path, index=False)
    print(f"Saved per-feature CMI detail to {detail_path}")

    # Summary: mean deviation per (dataset, mechanism, rate).
    summary = (
        merged[merged["train_initializer"] != "cold"]
        .groupby(["dataset", "mechanism", "miss_rate"])
        .agg(
            mean_cmi_deviation=("cmi_deviation", "mean"),
            std_cmi_deviation=("cmi_deviation", "std"),
            mean_cmi=("mean_cmi", "mean"),
        )
        .reset_index()
    )
    summary_path = args.output_dir / "cmi_bias_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved CMI bias summary to {summary_path}")

    # Per-feature baseline vs missing comparison (averaged over seeds).
    comparison = (
        merged.groupby(["dataset", "train_initializer", "feature_idx"])[
            "mean_cmi"
        ]
        .mean()
        .reset_index()
        .pivot_table(
            index=["dataset", "feature_idx"],
            columns="train_initializer",
            values="mean_cmi",
        )
    )
    comparison_path = args.output_dir / "cmi_feature_comparison.csv"
    comparison.to_csv(comparison_path)
    print(f"Saved feature comparison to {comparison_path}")

    print("Done.")


if __name__ == "__main__":
    main()
