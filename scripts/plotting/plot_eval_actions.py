from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from afabench.evaluation.tables import (
    combine_soft_budget_params,
    load_plotting_ready_table,
)
from afabench.plotting.config import (
    PlotEvalActionsConfig,
    PlottingDisplayConfig,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from matplotlib.axes import Axes

type Heatmap = Any

PLOT_FONT_SIZE = 16
PLOT_TITLE_FONT_SIZE = 18


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

    evaluation_df = load_plotting_ready_table(Path(cfg.input))
    evaluation_df = combine_soft_budget_params(evaluation_df)

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
