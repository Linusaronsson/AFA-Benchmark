from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from afabench.plotting.config import (
    PlotEvalActionsConfig,
    PlottingDisplayConfig,
)
from scripts.plotting.plot_eval_actions import (
    main,
    normalize_heatmap_by_timestep,
    read_parquet,
)


def action_row(
    method: str,
    dataset: str,
    *,
    action: int,
    n_selections: int,
    hard_budget: float | None = None,
    train_soft_budget_param: float | None = None,
    eval_soft_budget_param: float | None = None,
) -> dict[str, object]:
    return {
        "action_performed": action,
        "n_selections_performed": n_selections,
        "eval_hard_budget": hard_budget,
        "train_soft_budget_param": train_soft_budget_param,
        "eval_soft_budget_param": eval_soft_budget_param,
        "afa_method": method,
        "dataset": dataset,
    }


ACTION_SCHEMA = pa.schema(
    {
        "action_performed": pa.uint64(),
        "n_selections_performed": pa.uint64(),
        "eval_hard_budget": pa.float64(),
        "train_soft_budget_param": pa.float64(),
        "eval_soft_budget_param": pa.float64(),
        "afa_method": pa.string(),
        "dataset": pa.string(),
    }
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> Path:
    pq.write_table(pa.Table.from_pylist(rows, schema=ACTION_SCHEMA), path)
    return path


def test_heatmap_is_normalized_by_instances_at_each_time_step(
    tmp_path: Path,
) -> None:
    path = write_rows(
        tmp_path / "actions.parquet",
        [
            action_row("jafa", "cube", action=2, n_selections=0),
            action_row("jafa", "cube", action=2, n_selections=0),
            action_row("jafa", "cube", action=1, n_selections=0),
            action_row("jafa", "cube", action=1, n_selections=1),
        ],
    )

    heatmap = normalize_heatmap_by_timestep(
        read_parquet(path), max_action=2, max_time=1
    )

    # Rows are actions 1 and 2, columns are time steps 0 and 1
    np.testing.assert_allclose(heatmap, [[1 / 3, 1.0], [2 / 3, 0.0]])


def test_one_heatmap_file_per_dataset_and_budget(
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    input_path = write_rows(
        tmp_path / "actions.parquet",
        [
            # Hard budget: all methods share one figure per dataset & budget
            action_row("jafa", "cube", action=2, n_selections=0, hard_budget=3.0),
            action_row("jafa", "cube", action=0, n_selections=1, hard_budget=3.0),
            action_row("gdfs", "cube", action=1, n_selections=0, hard_budget=3.0),
            # Soft budget: one figure per method, whichever parameter is set
            action_row(
                "gdfs", "mnist", action=3, n_selections=0,
                train_soft_budget_param=0.1,
            ),
            action_row(
                "jafa", "mnist", action=1, n_selections=0,
                eval_soft_budget_param=0.5,
            ),
            # Only stop actions: nothing to plot
            action_row(
                "jafa", "mnist", action=0, n_selections=0,
                eval_soft_budget_param=0.7,
            ),
        ],
    )  # fmt: skip
    output_folder = tmp_path / "plots"
    cfg = OmegaConf.structured(
        PlotEvalActionsConfig(
            input=str(input_path),
            output_folder=str(output_folder),
            formats=["png"],
            plotting=plotting_config,
        )
    )

    main(cfg)

    assert sorted(
        path.relative_to(output_folder).as_posix()
        for path in output_folder.rglob("*.png")
    ) == [
        "hard_budget/cube/cube_action_heatmap_3.0.png",
        "soft_budget/mnist/mnist_gdfs_action_heatmap_0.1.png",
        "soft_budget/mnist/mnist_jafa_action_heatmap_0.5.png",
    ]
