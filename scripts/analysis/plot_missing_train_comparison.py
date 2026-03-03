# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""
Generate thesis-ready plots for CMI missing-train experiments.

Reads the accuracy_summary.csv and gap_to_baseline.csv produced by
missing_train_analysis.py and generates publication-quality figures.

Usage:
    python scripts/analysis/plot_missing_train_comparison.py \
        --input-dir extra/output/analysis/missing_train \
        --output-dir extra/output/analysis/missing_train/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Method display names.
METHOD_NAMES = {
    "gadgil2023": "DIME",
    "ma2018_external": "EDDI",
    "covert2023": "GDFS",
}

# Colors for mechanisms.
MECHANISM_COLORS = {
    "MCAR": "#1f77b4",
    "MAR": "#ff7f0e",
    "MNAR": "#2ca02c",
}

# Line styles for missingness rates.
RATE_STYLES = {
    0.1: "dotted",
    0.3: "dashed",
    0.5: "solid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("extra/output/analysis/missing_train"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extra/output/analysis/missing_train/figures"),
    )
    return parser.parse_args()


def plot_accuracy_vs_budget(
    acc: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Plot accuracy vs budget for each dataset."""
    datasets = sorted(acc["dataset"].unique())
    methods = sorted(acc["afa_method"].unique())

    for method in methods:
        method_label = METHOD_NAMES.get(method, method)
        df_method = acc[acc["afa_method"] == method]
        n_datasets = len(datasets)

        fig, axes = plt.subplots(
            1, n_datasets, figsize=(5 * n_datasets, 4), sharey=True
        )
        if n_datasets == 1:
            axes = [axes]

        for ax, dataset in zip(axes, datasets, strict=False):
            df_ds = df_method[df_method["dataset"] == dataset]

            # Plot baseline.
            baseline = df_ds[df_ds["train_initializer"] == "cold"]
            if not baseline.empty:
                ax.plot(
                    baseline["eval_hard_budget"],
                    baseline["accuracy"],
                    color="black",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label="Baseline (full data)",
                    zorder=10,
                )

            # Plot each (mechanism, rate) combination.
            for init_name, group in df_ds.groupby("train_initializer"):
                if init_name == "cold":
                    continue
                mechanism = group["mechanism"].iloc[0]
                rate = group["miss_rate"].iloc[0]
                color = MECHANISM_COLORS.get(mechanism, "gray")
                style = RATE_STYLES.get(rate, "solid")
                label = f"{mechanism} p={rate}"
                ax.plot(
                    group["eval_hard_budget"],
                    group["accuracy"],
                    color=color,
                    linestyle=style,
                    marker="s",
                    markersize=3,
                    alpha=0.8,
                    label=label,
                )

            ax.set_title(f"{dataset}")
            ax.set_xlabel("Hard Budget")
            if ax == axes[0]:
                ax.set_ylabel("Accuracy")
            ax.grid(alpha=0.3)

        # Single legend for the figure.
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(5, len(labels)),
            bbox_to_anchor=(0.5, -0.15),
            fontsize=8,
        )
        fig.suptitle(
            f"{method_label}: Accuracy vs Budget Under Missing Training Data"
        )
        fig.tight_layout()
        path = output_dir / f"accuracy_vs_budget_{method}.pdf"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved {path}")


def plot_gap_heatmap(gap: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap: method x mechanism showing mean gap-to-baseline."""
    non_baseline = gap[gap["train_initializer"] != "cold"]
    if non_baseline.empty:
        print("No non-baseline data for heatmap.")
        return

    pivot = (
        non_baseline.groupby(["afa_method", "mechanism"])["gap"]
        .mean()
        .reset_index()
    )
    pivot["afa_method"] = pivot["afa_method"].map(
        lambda x: METHOD_NAMES.get(x, x)
    )
    pivot_table = pivot.pivot_table(
        index="afa_method", columns="mechanism", values="gap"
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        ax=ax,
    )
    ax.set_title("Mean Accuracy Gap to Baseline (method x mechanism)")
    ax.set_ylabel("")
    fig.tight_layout()
    path = output_dir / "gap_heatmap.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_mechanism_sensitivity(gap: pd.DataFrame, output_dir: Path) -> None:
    """Box plots: for each method, accuracy gap distribution by mechanism."""
    non_baseline = gap[gap["train_initializer"] != "cold"].copy()
    if non_baseline.empty:
        print("No non-baseline data for sensitivity plot.")
        return

    non_baseline["method_label"] = non_baseline["afa_method"].map(
        lambda x: METHOD_NAMES.get(x, x)
    )
    methods = sorted(non_baseline["method_label"].unique())
    n_methods = len(methods)

    fig, axes = plt.subplots(
        1, n_methods, figsize=(4 * n_methods, 4), sharey=True
    )
    if n_methods == 1:
        axes = [axes]

    for ax, method in zip(axes, methods, strict=False):
        df_m = non_baseline[non_baseline["method_label"] == method]
        sns.boxplot(
            data=df_m,
            x="mechanism",
            y="gap",
            order=["MCAR", "MAR", "MNAR"],
            palette=MECHANISM_COLORS,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(method)
        ax.set_xlabel("Mechanism")
        if ax == axes[0]:
            ax.set_ylabel("Accuracy Gap to Baseline")
        else:
            ax.set_ylabel("")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Sensitivity of CMI Methods to Missingness Mechanism")
    fig.tight_layout()
    path = output_dir / "mechanism_sensitivity.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def generate_gap_table(gap: pd.DataFrame, output_dir: Path) -> None:
    """LaTeX-ready table of accuracy degradation."""
    non_baseline = gap[gap["train_initializer"] != "cold"].copy()
    if non_baseline.empty:
        return

    non_baseline["method_label"] = non_baseline["afa_method"].map(
        lambda x: METHOD_NAMES.get(x, x)
    )

    table = (
        non_baseline.groupby(
            ["method_label", "dataset", "mechanism", "miss_rate"]
        )["gap"]
        .mean()
        .reset_index()
        .pivot_table(
            index=["method_label", "dataset"],
            columns=["mechanism", "miss_rate"],
            values="gap",
        )
    )
    table_path = output_dir / "gap_table.csv"
    table.to_csv(table_path)
    print(f"Saved gap table to {table_path}")

    # Also save LaTeX version.
    latex_path = output_dir / "gap_table.tex"
    table.to_latex(latex_path, float_format="%.3f", multirow=True)
    print(f"Saved LaTeX table to {latex_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    acc_path = args.input_dir / "accuracy_summary.csv"
    gap_path = args.input_dir / "gap_to_baseline.csv"

    if not acc_path.exists() or not gap_path.exists():
        print(
            f"Input files not found. Run missing_train_analysis.py first.\n"
            f"  Expected: {acc_path}\n"
            f"  Expected: {gap_path}"
        )
        return

    acc = pd.read_csv(acc_path)
    gap = pd.read_csv(gap_path)

    print(f"Loaded {len(acc)} accuracy rows, {len(gap)} gap rows.")

    plot_accuracy_vs_budget(acc, args.output_dir)
    plot_gap_heatmap(gap, args.output_dir)
    plot_mechanism_sensitivity(gap, args.output_dir)
    generate_gap_table(gap, args.output_dir)

    print("All plots generated.")


if __name__ == "__main__":
    main()
