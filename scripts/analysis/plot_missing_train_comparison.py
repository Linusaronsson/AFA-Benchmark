# pyright: reportCallIssue=false, reportAttributeAccessIssue=false, reportArgumentType=false
"""
Generate thesis-ready plots for train-missing experiments.

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
    "aaco_full": "AACO full",
    "aaco_zero_fill": "AACO zero-fill",
    "aaco_mask_aware": "AACO mask-aware",
    "aaco_dr": "AACO-DR",
    "gadgil2023": "DIME + block-only",
    "gadgil2023_ipw_feature_marginal": "DIME + IPW",
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

# Mitigation families: group methods by the base method they mitigate.
METHOD_FAMILIES: dict[str, list[str]] = {
    "AACO": ["aaco_full", "aaco_zero_fill", "aaco_mask_aware", "aaco_dr"],
    "DIME": ["gadgil2023", "gadgil2023_ipw_feature_marginal"],
}

# Per-method colors within a family (Dark2 palette for consistency).
MITIGATION_COLORS: dict[str, str] = {
    "aaco_full": "#1b9e77",
    "aaco_zero_fill": "#d95f02",
    "aaco_mask_aware": "#7570b3",
    "aaco_dr": "#e7298a",
    "gadgil2023": "#1b9e77",
    "gadgil2023_ipw_feature_marginal": "#d95f02",
}

MITIGATION_MARKERS: dict[str, str] = {
    "aaco_full": "^",
    "aaco_zero_fill": "D",
    "aaco_mask_aware": "P",
    "aaco_dr": "h",
    "gadgil2023": "o",
    "gadgil2023_ipw_feature_marginal": "s",
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


def plot_mitigation_comparison(  # noqa: C901
    acc: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Compare mitigation variants within each method family.

    For each (dataset, mechanism, miss_rate) condition, create one figure
    with one subplot per method family.  Each subplot shows accuracy vs
    budget for every mitigation variant, plus a dashed black baseline from
    the cold (full-data) run.
    """
    non_cold = acc[acc["train_initializer"] != "cold"]
    if non_cold.empty:
        print("No non-baseline data for mitigation comparison.")
        return

    cold = acc[acc["train_initializer"] == "cold"]
    datasets = sorted(non_cold["dataset"].unique())

    conditions = (
        non_cold[["mechanism", "miss_rate"]]
        .drop_duplicates()
        .sort_values(["mechanism", "miss_rate"])
    )

    for dataset in datasets:
        for _, cond_row in conditions.iterrows():
            mechanism = cond_row["mechanism"]
            miss_rate = cond_row["miss_rate"]
            df_cond = non_cold[
                (non_cold["dataset"] == dataset)
                & (non_cold["mechanism"] == mechanism)
                & (non_cold["miss_rate"] == miss_rate)
            ]
            if df_cond.empty:
                continue

            # Only keep families with >=2 members present.
            active_families = {}
            for family_name, members in METHOD_FAMILIES.items():
                present = [
                    m for m in members if m in df_cond["afa_method"].unique()
                ]
                if len(present) >= 2:
                    active_families[family_name] = present

            if not active_families:
                continue

            n_fam = len(active_families)
            fig, axes = plt.subplots(
                1,
                n_fam,
                figsize=(5 * n_fam, 4),
                sharey=True,
                squeeze=False,
            )

            for ax, (fam_name, members) in zip(
                axes[0], active_families.items(), strict=False
            ):
                # Baseline from cold for the first family member.
                baseline = cold[
                    (cold["dataset"] == dataset)
                    & (cold["afa_method"].isin(members))
                ]
                if not baseline.empty:
                    bl_method = baseline["afa_method"].iloc[0]
                    bl = baseline[
                        baseline["afa_method"] == bl_method
                    ].sort_values("eval_hard_budget")
                    ax.plot(
                        bl["eval_hard_budget"],
                        bl["accuracy"],
                        color="black",
                        linewidth=1.5,
                        linestyle="--",
                        marker="o",
                        markersize=3,
                        label="Full data",
                        zorder=10,
                    )

                for member in members:
                    df_m = df_cond[
                        df_cond["afa_method"] == member
                    ].sort_values("eval_hard_budget")
                    if df_m.empty:
                        continue
                    label = METHOD_NAMES.get(member, member)
                    color = MITIGATION_COLORS.get(member, "gray")
                    marker = MITIGATION_MARKERS.get(member, "o")
                    ax.plot(
                        df_m["eval_hard_budget"],
                        df_m["accuracy"],
                        color=color,
                        marker=marker,
                        markersize=4,
                        linewidth=1.5,
                        label=label,
                    )

                ax.set_title(fam_name)
                ax.set_xlabel("Hard Budget")
                if ax == axes[0, 0]:
                    ax.set_ylabel("Accuracy")
                ax.legend(fontsize=7, loc="lower right")
                ax.grid(alpha=0.3)

            rate_str = f"{miss_rate:.1f}".replace(".", "")
            fig.suptitle(f"{dataset}: {mechanism} p={miss_rate}")
            fig.tight_layout()
            fname = f"mitigation_{dataset}_{mechanism}_{rate_str}.pdf"
            path = output_dir / fname
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
    plot_mitigation_comparison(acc, args.output_dir)
    generate_gap_table(gap, args.output_dir)

    print("All plots generated.")


if __name__ == "__main__":
    main()
