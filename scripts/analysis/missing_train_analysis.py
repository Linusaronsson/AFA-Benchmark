# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportAssignmentType=false
"""
Aggregate and summarize CMI missing-train experiment results.

Reads merged parquet files from the pipeline output for each train_initializer
condition, computes accuracy metrics per method/dataset/mechanism/rate, and
generates summary tables.

Usage:
    python scripts/analysis/missing_train_analysis.py \
        --results-dir extra/output/merged_results \
        --eval-split val \
        --output-dir extra/output/analysis/missing_train
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Map initializer names to (mechanism, rate) tuples.
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
        default=Path("extra/output/merged_results"),
    )
    parser.add_argument("--eval-split", default="val")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extra/output/analysis/missing_train"),
    )
    parser.add_argument(
        "--method-set",
        default="cmi_missing_train",
        help="Method set name in the merged parquet filename.",
    )
    return parser.parse_args()


def load_all_results(
    results_dir: Path,
    eval_split: str,
    method_set: str,
) -> pd.DataFrame:
    """Load merged parquets for all train initializers."""
    frames = []
    split_dir = results_dir / f"eval_split-{eval_split}"
    if not split_dir.exists():
        msg = f"No results directory at {split_dir}"
        raise FileNotFoundError(msg)

    for init_dir in sorted(split_dir.iterdir()):
        if not init_dir.is_dir():
            continue
        # Extract initializer name from directory (e.g. "train_init-mcar_p03+eval_init-cold")
        init_name = _extract_train_initializer(init_dir.name)
        if init_name is None or init_name not in INITIALIZER_META:
            continue

        parquet_path = (
            init_dir / "eval_perf" / f"method_set-{method_set}+all.parquet"
        )
        if not parquet_path.exists():
            print(f"Warning: missing {parquet_path}, skipping.")
            continue

        df = pd.read_parquet(parquet_path)
        mechanism, rate = INITIALIZER_META[init_name]
        df["train_initializer"] = init_name
        df["mechanism"] = mechanism
        df["miss_rate"] = rate
        frames.append(df)

    if not frames:
        msg = f"No parquet files found under {split_dir} for method_set={method_set}"
        raise FileNotFoundError(msg)
    return pd.concat(frames, ignore_index=True)


def _extract_train_initializer(dirname: str) -> str | None:
    """Extract train initializer name from a directory name like 'train_init-mcar_p03+eval_init-cold'."""
    match = re.search(r"train_init-([^+]+)", dirname)
    if match:
        return match.group(1)
    # Fallback: the directory name might just be the initializer name.
    if dirname in INITIALIZER_META:
        return dirname
    return None


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute accuracy at each (method, dataset, budget, initializer) combination.

    Accuracy is computed at the final timestep for each sample (the last row
    per idx within a budget group).
    """
    # Filter to external classifier predictions if available, else builtin.
    if "classifier" in df.columns:
        df_ext = df[df["classifier"] == "external"]
        if df_ext.empty:
            df_ext = df[df["classifier"] == "builtin"]
        df = df_ext

    # Keep only the last step per sample (max n_selections_performed per idx).
    idx_cols = [
        "afa_method",
        "dataset",
        "eval_hard_budget",
        "train_initializer",
        "eval_seed",
        "idx",
    ]
    available_cols = [c for c in idx_cols if c in df.columns]
    df_last = df.loc[
        df.groupby(available_cols)["n_selections_performed"].idxmax()
    ]

    df_last = df_last.copy()
    df_last["correct"] = (
        df_last["predicted_class"] == df_last["true_class"]
    ).astype(int)

    group_cols = [
        "afa_method",
        "dataset",
        "eval_hard_budget",
        "train_initializer",
        "mechanism",
        "miss_rate",
    ]
    available_group = [c for c in group_cols if c in df_last.columns]

    acc = (
        df_last.groupby(available_group)["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    return acc


def compute_gap_to_baseline(acc: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy gap relative to the cold (full-data) baseline."""
    baseline = acc[acc["train_initializer"] == "cold"][
        ["afa_method", "dataset", "eval_hard_budget", "accuracy"]
    ].rename(columns={"accuracy": "baseline_accuracy"})

    merged = acc.merge(
        baseline,
        on=["afa_method", "dataset", "eval_hard_budget"],
        how="left",
    )
    merged["gap"] = merged["accuracy"] - merged["baseline_accuracy"]
    return merged


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_all_results(args.results_dir, args.eval_split, args.method_set)
    print(
        f"Loaded {len(df)} rows across {df['train_initializer'].nunique()} initializers."
    )

    print("Computing accuracy...")
    acc = compute_accuracy(df)

    print("Computing gap to baseline...")
    gap = compute_gap_to_baseline(acc)

    # Save detailed results.
    acc_path = args.output_dir / "accuracy_summary.csv"
    acc.to_csv(acc_path, index=False)
    print(f"Saved accuracy summary to {acc_path}")

    gap_path = args.output_dir / "gap_to_baseline.csv"
    gap.to_csv(gap_path, index=False)
    print(f"Saved gap-to-baseline to {gap_path}")

    # Mean accuracy across budgets.
    mean_acc = (
        acc.groupby(
            [
                "afa_method",
                "dataset",
                "train_initializer",
                "mechanism",
                "miss_rate",
            ]
        )["accuracy"]
        .mean()
        .reset_index()
        .rename(columns={"accuracy": "mean_accuracy"})
    )
    mean_acc_path = args.output_dir / "mean_accuracy_across_budgets.csv"
    mean_acc.to_csv(mean_acc_path, index=False)
    print(f"Saved mean accuracy to {mean_acc_path}")

    # Pivot heatmap: method x mechanism, averaged across datasets and rates.
    heatmap_data = (
        gap[gap["train_initializer"] != "cold"]
        .groupby(["afa_method", "mechanism"])["gap"]
        .mean()
        .reset_index()
        .pivot_table(index="afa_method", columns="mechanism", values="gap")
    )
    heatmap_path = args.output_dir / "heatmap_method_x_mechanism.csv"
    heatmap_data.to_csv(heatmap_path)
    print(f"Saved heatmap data to {heatmap_path}")

    print("Done.")


if __name__ == "__main__":
    main()
