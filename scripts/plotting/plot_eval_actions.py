from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from afabench.plotting.config import (
    PlotEvalActionsConfig,
    PlottingDisplayConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

type Heatmap = Any

PLOT_FONT_SIZE = 16
PLOT_TITLE_FONT_SIZE = 18


def create_dummy_data() -> pd.DataFrame:
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

    return pd.DataFrame(rows).astype(
        {
            "action_performed": "UInt64",
            "true_class": "UInt64",
            "accumulated_cost": "Float64",
            "idx": "UInt64",
            "forced_stop": "boolean",
            "eval_seed": "UInt64",
            "eval_hard_budget": "Float64",
            "train_soft_budget_param": "Float64",
            "eval_soft_budget_param": "Float64",
            "n_selections_performed": "UInt64",
            "afa_method": "string",
            "dataset": "string",
            "train_seed": "UInt64",
            "train_hard_budget": "Float64",
            "predicted_class": "Int64",
        }
    )


def read_parquet(input_path: Path) -> pd.DataFrame:
    return pd.read_parquet(input_path, dtype_backend="numpy_nullable")


def assert_only_one_soft_budget_param_type(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure only one type of soft budget parameter is set."""
    assert (
        dataframe["train_soft_budget_param"].isna()
        | dataframe["eval_soft_budget_param"].isna()
    ).all(), (
        "Both train_soft_budget_param and eval_soft_budget_param"
        " cannot be set. Choose one."
    )
    return dataframe.assign(
        soft_budget_param=dataframe["train_soft_budget_param"].fillna(
            dataframe["eval_soft_budget_param"]
        )
    ).drop(columns=["train_soft_budget_param", "eval_soft_budget_param"])


def filter_only_largest_budget(dataframe: pd.DataFrame) -> pd.DataFrame:
    """For each dataset, only keep the largest evaluation budget."""
    largest_budget = dataframe.groupby("dataset")[
        "eval_hard_budget"
    ].transform("max")
    return dataframe.loc[
        (dataframe["eval_hard_budget"] == largest_budget).fillna(False)
    ]


def filter_only_smallest_soft_budget_parameter(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """For each dataset and method, only keep the smallest soft budget parameter."""
    smallest_parameter = dataframe.groupby(["afa_method", "dataset"])[
        "soft_budget_param"
    ].transform("min")
    return dataframe.loc[
        (dataframe["soft_budget_param"] == smallest_parameter).fillna(False)
    ]


def normalize_heatmap_by_timestep(
    df_method: pd.DataFrame,
    max_action: int,
    max_time: int,
) -> Heatmap:  # type: ignore[no-any-return]
    """
    Create and normalize heatmap for a single method.

    Heatmap shape: [num_actions, num_timesteps]
    Values normalized by number of samples at each timestep.
    Excludes action 0.
    """
    heatmap = np.zeros((int(max_action), int(max_time) + 1))  # type: ignore[arg-type]

    actions = df_method["action_performed"].to_numpy(dtype=np.int64)
    time_steps = df_method["n_selections_performed"].to_numpy(dtype=np.int64)
    # Skip action 0
    taken = actions > 0
    np.add.at(heatmap, (actions[taken] - 1, time_steps[taken]), 1)

    time_counts = np.bincount(
        time_steps,
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
    plotting_config: PlottingDisplayConfig,
) -> None:
    """Format and label axes for a heatmap subplot."""
    method_name = plotting_config.method_name_mapping.get(method, method)
    ax.set_title(
        method_name,
        fontsize=PLOT_FONT_SIZE,
        fontweight="bold",
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Action index")
    ax.set_xticks(np.arange(0, int(max_time) + 1, max(1, int(max_time) // 5)))
    max_action_int = int(max_action)
    y_ticks = np.arange(
        0, max_action_int + 1, max(1, (max_action_int + 1) // 10)
    )
    ax.set_yticks(y_ticks)
    # Increment y tick labels by 1 to show action numbers (1, 2, 3...) instead of (0, 1, 2...)
    ax.set_yticklabels(y_ticks + 1)


def produce_separate_plots(
    df: pd.DataFrame,
    output_folder: Path,
    budget_type: str,
    plotting_config: PlottingDisplayConfig,
    formats: Sequence[str] = ("pdf",),
) -> None:
    """
    Create separate plots for each method/dataset/budget combination.

    Args:
        df: Input dataframe
        output_folder: Output folder for plots
        budget_type: Type of budget ("hard_budget" or "soft_budget")
        formats: Output formats (e.g. ("pdf", "svg")). Default: ("pdf",)
    """
    # Filter out action 0
    filtered_df = df.loc[df["action_performed"] != 0]
    # Group by dataset and budget (and method for soft budget)
    if budget_type == "hard_budget":
        group_cols = ["dataset", "eval_hard_budget"]
    else:
        group_cols = ["dataset", "afa_method", "soft_budget_param"]

    for group_keys, group_df in tqdm(
        filtered_df.groupby(group_cols, dropna=False),
        desc=f"Creating separate {budget_type} plots",
    ):
        if budget_type == "hard_budget":
            dataset_name, hard_budget = group_keys
            extra_title = f"Hard Budget: {hard_budget}"
            filename_suffix = f"_{hard_budget}"
            methods = sorted(group_df["afa_method"].unique())
            method_filename = ""
        else:
            dataset_name, method_name, soft_budget_param = group_keys
            extra_title = f"Soft Budget Param: {soft_budget_param}"
            filename_suffix = f"_{soft_budget_param}"
            methods = [method_name]
            method_filename = f"_{method_name.replace('/', '_')}"

        # Create subdirectory for this dataset
        dataset_output_folder = output_folder / dataset_name
        dataset_output_folder.mkdir(parents=True, exist_ok=True)

        # Create the heatmap plot with 5 columns
        num_methods = len(methods)
        num_cols = 5
        num_rows = (num_methods + num_cols - 1) // num_cols

        # Calculate global max_action and max_time across all methods in group
        global_max_action = cast("int", group_df["action_performed"].max())
        global_max_time = cast("int", group_df["n_selections_performed"].max())

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(3 * num_cols, 4 * num_rows),
            squeeze=False,
        )

        for idx, method in enumerate(methods):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]
            df_method = group_df.loc[group_df["afa_method"] == method]

            heatmap = normalize_heatmap_by_timestep(
                df_method, global_max_action, global_max_time
            )

            ax.imshow(
                heatmap,
                cmap="Blues",
                aspect="auto",
                origin="lower",
                vmin=0.0,
                vmax=1.0,
            )

            format_heatmap_axes(
                ax, global_max_action, global_max_time, method, plotting_config
            )

        # Hide any unused subplots
        for idx in range(num_methods, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].set_visible(False)

        dataset_display_name = plotting_config.dataset_name_mapping.get(
            dataset_name, dataset_name
        )
        fig.suptitle(
            f"Action Heatmaps - {dataset_display_name} - {extra_title}",
            fontsize=PLOT_TITLE_FONT_SIZE,
            y=0.98,
        )
        plt.subplots_adjust(
            left=0.08, right=0.92, top=0.84, bottom=0.1, wspace=0.3
        )

        for fmt in formats:
            output_path = (
                dataset_output_folder
                / f"{dataset_name}{method_filename}_action_heatmap{filename_suffix}.{fmt}"
            )
            fig.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close(fig)


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/plotting/plot_eval_actions",
    config_name="config",
)
def main(cfg: PlotEvalActionsConfig) -> None:
    cfg = cast("PlotEvalActionsConfig", OmegaConf.to_object(cfg))
    assert isinstance(cfg, PlotEvalActionsConfig)

    plt.rcParams["font.family"] = cfg.plotting.plot_font_family
    plt.rcParams["font.size"] = PLOT_FONT_SIZE

    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    evaluation_df = read_parquet(Path(cfg.input))
    evaluation_df = assert_only_one_soft_budget_param_type(evaluation_df)

    hard_budget_folder = output_folder / "hard_budget"
    soft_budget_folder = output_folder / "soft_budget"

    hard_budget_folder.mkdir(parents=True, exist_ok=True)
    soft_budget_folder.mkdir(parents=True, exist_ok=True)

    # Filter by hard budget (all combinations)
    evaluation_df_hard_budget = evaluation_df.loc[
        evaluation_df["eval_hard_budget"].notna()
    ]
    # Filter by soft budget (all combinations)
    evaluation_df_soft_budget = evaluation_df.loc[
        evaluation_df["soft_budget_param"].notna()
    ]

    produce_separate_plots(
        df=evaluation_df_hard_budget,
        output_folder=hard_budget_folder,
        budget_type="hard_budget",
        plotting_config=cfg.plotting,
        formats=cfg.formats,
    )
    produce_separate_plots(
        df=evaluation_df_soft_budget,
        output_folder=soft_budget_folder,
        budget_type="soft_budget",
        plotting_config=cfg.plotting,
        formats=cfg.formats,
    )


if __name__ == "__main__":
    main()
