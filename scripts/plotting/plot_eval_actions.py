from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from tqdm import tqdm

if TYPE_CHECKING:
    from matplotlib.axes import Axes

type Heatmap = Any

PLOT_WIDTH = 13
PLOT_HEIGHT = 5

METHOD_NAME_MAPPING = {
    # Dummy
    "random_dummy": "Random dummy",
    "sequential_dummy": "Sequential dummy",
    # RL
    "jafa": "JAFA",
    "odin_model_based": "ODIN-MB",
    "odin_model_free": "ODIN-MF",
    "ol_without_mask": "OL",
    "ol_with_mask": "OL+mask",
    "eddi": "EDDI",
    "dime": "DIME",
    "aaco": "AACO",
    "aaco_nn": "AACO+NN",
}

DATASET_NAME_MAPPING = {
    "cube": "Cube",
    "afa_context": "AFAContext",
    "synthetic_mnist": "Synthetic MNIST",
    "cube_without_noise": "Noiseless Cube",
    "afa_context_without_noise": "Noiseless AFAContext",
    "synthetic_mnist_without_noise": "Noiseless Synthetic MNIST",
    "mnist": "MNIST",
    "actg": "ACTG",
    "bank_marketing": "Bank Marketing",
    "ckd": "CKD",
    "diabetes": "Diabetes",
    "fashion_mnist": "FashionMNIST",
    "miniboone": "MiniBooNE",
    "physionet": "PhysioNet",
}
DATASETS = DATASET_NAME_MAPPING.keys()


def create_dummy_data() -> pl.DataFrame:
    """Create minimal dummy data for testing action heatmap plots."""
    methods = ["jafa", "odin_model_based", "random_dummy"]
    datasets_list = ["cube_without_noise", "synthetic_mnist_without_noise"]
    train_seeds = [0, 1]
    eval_seeds = [0, 1]
    hard_budgets = [5.0]
    num_samples = 100
    rng = np.random.default_rng(42)

    # Hard budget data with multiple actions per sample
    rows = [
        {
            "action_performed": int(rng.integers(0, 10)),
            "true_class": int(rng.integers(0, 3)),
            "accumulated_cost": float(budget + rng.normal(0, 0.1)),
            "idx": sample_idx,
            "forced_stop": False,
            "eval_seed": eval_seed,
            "eval_hard_budget": float(budget),
            "train_soft_budget_param": None,
            "eval_soft_budget_param": None,
            "n_selections_performed": action_idx + 1,
            "afa_method": method,
            "dataset": dataset,
            "train_seed": train_seed,
            "train_hard_budget": None,
            "predicted_class": int(rng.integers(0, 3)),
        }
        for method in methods
        for dataset in datasets_list
        for train_seed in train_seeds
        for eval_seed in eval_seeds
        for budget in hard_budgets
        for sample_idx in range(num_samples)
        for action_idx in range(1, int(rng.integers(1, 8)) + 1)
    ]

    return pl.DataFrame(
        rows,
        schema={
            "action_performed": pl.UInt64,
            "true_class": pl.UInt64,
            "accumulated_cost": pl.Float64,
            "idx": pl.UInt64,
            "forced_stop": pl.Boolean,
            "eval_seed": pl.UInt64,
            "eval_hard_budget": pl.Float64,
            "train_soft_budget_param": pl.Float64,
            "eval_soft_budget_param": pl.Float64,
            "n_selections_performed": pl.UInt64,
            "afa_method": pl.String,
            "dataset": pl.String,
            "train_seed": pl.UInt64,
            "train_hard_budget": pl.Float64,
            "predicted_class": pl.Int64,
        },
    )


def read_parquet(input_path: Path) -> pl.DataFrame:
    return pl.read_parquet(input_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation action heatmaps from results.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input Parquet file with evaluation results.",
    )
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Output folder where plots will be saved",
    )
    return parser.parse_args()


def assert_only_one_soft_budget_param_type(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Ensure only one type of soft budget parameter is set."""
    assert (
        dataframe["train_soft_budget_param"].is_null()
        | dataframe["eval_soft_budget_param"].is_null()
    ).all(), (
        "Both train_soft_budget_param and eval_soft_budget_param"
        " cannot be set. Choose one."
    )
    dataframe = dataframe.with_columns(
        soft_budget_param=pl.coalesce(
            "train_soft_budget_param", "eval_soft_budget_param"
        )
    ).drop(["train_soft_budget_param", "eval_soft_budget_param"])
    return dataframe


def filter_only_largest_budget(dataframe: pl.DataFrame) -> pl.DataFrame:
    """For each dataset, only keep the largest evaluation budget."""
    return dataframe.filter(
        pl.col("eval_hard_budget")
        == pl.col("eval_hard_budget").max().over("dataset")
    )


def normalize_heatmap_by_timestep(
    df_method: pl.DataFrame,
    max_action: int,
    max_time: int,
) -> Heatmap:  # type: ignore[no-any-return]
    """
    Create and normalize heatmap for a single method.

    Heatmap shape: [num_actions, num_timesteps]
    Values normalized by number of samples at each timestep.
    """
    heatmap = np.zeros((int(max_action) + 1, int(max_time) + 1))  # type: ignore[arg-type]

    for row in df_method.iter_rows(named=True):
        action = int(row["action_performed"])
        time_step = int(row["n_selections_performed"])
        # if 0 <= action <= int(max_action) and 0 <= time_step <= int(max_time):
        heatmap[action, time_step] += 1

    time_counts = np.bincount(
        df_method["n_selections_performed"].cast(pl.Int64).to_numpy(),
        minlength=int(max_time) + 1,
    )
    time_counts = np.maximum(time_counts, 1)
    heatmap = heatmap / time_counts

    return heatmap


def format_heatmap_axes(
    ax: Axes,
    max_action: float,
    max_time: float,
    method: str,
) -> None:
    """Format and label axes for a heatmap subplot."""
    method_name = METHOD_NAME_MAPPING.get(method, method)
    ax.set_title(method_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Action index")
    ax.set_xticks(np.arange(0, int(max_time) + 1, max(1, int(max_time) // 5)))
    max_action_int = int(max_action)
    ax.set_yticks(
        np.arange(0, max_action_int + 1, max(1, (max_action_int + 1) // 10))
    )


def create_action_heatmap(
    dataframe: pl.DataFrame, dataset: str, output_folder: Path
) -> None:
    """
    Create action heatmaps for all methods in a dataset.

    One figure per dataset with subplots for each method.
    X-axis: time (n_selections_performed)
    Y-axis: action index
    """
    methods = sorted(dataframe["afa_method"].unique())
    num_methods = len(methods)

    fig, axes = plt.subplots(
        1, num_methods, figsize=(5 * num_methods, 4), squeeze=False
    )
    axes = axes[0]

    for idx, method in enumerate(methods):
        ax = axes[idx]
        df_method = dataframe.filter(pl.col("afa_method") == method)

        max_action = cast("int", df_method["action_performed"].max())
        max_time = cast("int", df_method["n_selections_performed"].max())

        heatmap = normalize_heatmap_by_timestep(
            df_method, max_action, max_time
        )

        ax.imshow(
            heatmap,
            cmap="Blues",
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )

        format_heatmap_axes(ax, max_action, max_time, method)

    dataset_name = DATASET_NAME_MAPPING.get(dataset, dataset)
    fig.suptitle(f"{dataset_name} - Action Heatmaps", fontsize=14, y=0.98)
    plt.subplots_adjust(
        left=0.08, right=0.92, top=0.84, bottom=0.1, wspace=0.3
    )

    output_path = output_folder / f"{dataset}_action_heatmap.pdf"
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)

    evaluation_df = read_parquet(args.input)
    evaluation_df = assert_only_one_soft_budget_param_type(evaluation_df)
    evaluation_df = filter_only_largest_budget(evaluation_df)

    # One plot per dataset
    for keys_tuple, dataset_df in tqdm(
        evaluation_df.group_by("dataset"), desc="Creating action heatmaps"
    ):
        dataset_name = keys_tuple[0]
        create_action_heatmap(dataset_df, dataset_name, args.output_folder)


if __name__ == "__main__":
    main()
