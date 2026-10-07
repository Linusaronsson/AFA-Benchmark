from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from afabench.plotting.config import PlottingDisplayConfig
from scripts.plotting.plot_eval_perf import EvaluationPlotter

# The (ddof=1) standard deviation of two values `d` apart is d * SQRT_HALF
SQRT_HALF = 0.5**0.5


def eval_row(
    method: str,
    dataset: str,
    train_seed: int,
    *,
    action: int,
    n_selections: int,
    true_class: int,
    predicted_class: int | None,
    cost: float,
    hard_budget: float | None = None,
    eval_soft_budget_param: float | None = None,
) -> dict[str, object]:
    return {
        "action_performed": action,
        "true_class": true_class,
        "accumulated_cost": cost,
        "forced_stop": False,
        "eval_seed": 0,
        "eval_hard_budget": hard_budget,
        "n_selections_performed": n_selections,
        "predicted_class": predicted_class,
        "afa_method": method,
        "dataset": dataset,
        "initializer": "cold",
        "train_seed": train_seed,
        "train_hard_budget": None,
        "train_soft_budget_param": None,
        "eval_soft_budget_param": eval_soft_budget_param,
    }


EVAL_SCHEMA = pa.schema(
    {
        "action_performed": pa.uint64(),
        "true_class": pa.uint64(),
        "accumulated_cost": pa.float64(),
        "forced_stop": pa.bool_(),
        "eval_seed": pa.uint64(),
        "eval_hard_budget": pa.float64(),
        "n_selections_performed": pa.uint64(),
        "predicted_class": pa.uint64(),
        "afa_method": pa.string(),
        "dataset": pa.string(),
        "initializer": pa.string(),
        "train_seed": pa.uint64(),
        "train_hard_budget": pa.float64(),
        "train_soft_budget_param": pa.float64(),
        "eval_soft_budget_param": pa.float64(),
    }
)


@pytest.fixture
def eval_perf_path(tmp_path: Path) -> Path:
    rows = [
        # JAFA on cube at hard budget 2, two training seeds. Seed 0 makes an
        # intermediate wrong guess, then stops with one of two instances right.
        eval_row(
            "jafa", "cube", 0, action=3, n_selections=0, true_class=1,
            predicted_class=0, cost=0.0, hard_budget=2.0,
        ),
        eval_row(
            "jafa", "cube", 0, action=0, n_selections=2, true_class=1,
            predicted_class=1, cost=2.0, hard_budget=2.0,
        ),
        eval_row(
            "jafa", "cube", 0, action=0, n_selections=1, true_class=0,
            predicted_class=1, cost=1.0, hard_budget=2.0,
        ),
        # Rows without a prediction are ignored
        eval_row(
            "jafa", "cube", 0, action=0, n_selections=1, true_class=0,
            predicted_class=None, cost=1.0, hard_budget=2.0,
        ),
        # Seed 1 gets both instances right
        eval_row(
            "jafa", "cube", 1, action=0, n_selections=2, true_class=1,
            predicted_class=1, cost=2.0, hard_budget=2.0,
        ),
        eval_row(
            "jafa", "cube", 1, action=0, n_selections=2, true_class=0,
            predicted_class=0, cost=2.0, hard_budget=2.0,
        ),
        # A smaller hard budget, evaluated with a single seed
        eval_row(
            "jafa", "cube", 0, action=0, n_selections=1, true_class=1,
            predicted_class=0, cost=1.0, hard_budget=1.0,
        ),
        # GDFS on mnist (an F1 dataset) under a soft budget, single seed
        eval_row(
            "gdfs", "mnist", 0, action=0, n_selections=3, true_class=1,
            predicted_class=1, cost=3.0, eval_soft_budget_param=0.1,
        ),
        eval_row(
            "gdfs", "mnist", 0, action=0, n_selections=3, true_class=0,
            predicted_class=1, cost=3.0, eval_soft_budget_param=0.1,
        ),
    ]  # fmt: skip
    path = tmp_path / "eval_perf.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=EVAL_SCHEMA), path)
    return path


def sorted_rows(df: object, *keys: str) -> list[dict[str, object]]:
    rows = pa.table(df).to_pylist()
    return sorted(rows, key=lambda row: tuple(str(row[key]) for key in keys))


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_stop_action_metrics_are_aggregated_over_seeds(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    plotter = EvaluationPlotter(eval_perf_path, tmp_path, plotting_config)

    plotter.load_and_process()

    assert plotter.df_stop_action is not None
    rows = sorted_rows(
        plotter.df_stop_action, "afa_method", "eval_hard_budget"
    )
    assert rows == [
        {
            "afa_method": "gdfs",
            "dataset": "mnist",
            "eval_hard_budget": None,
            "soft_budget_param": 0.1,
            "mean_accuracy": 0.5,
            "std_accuracy": None,
            "mean_f_score": pytest.approx(1 / 3),
            "std_f_score": None,
            "mean_avg_accumulated_cost": 3.0,
            "std_avg_accumulated_cost": None,
            # mnist is scored with F1
            "mean_metric": pytest.approx(1 / 3),
            "std_metric": None,
            "low_metric": None,
            "high_metric": None,
            "low_avg_accumulated_cost": None,
            "high_avg_accumulated_cost": None,
        },
        {
            "afa_method": "jafa",
            "dataset": "cube",
            "eval_hard_budget": 1.0,
            "soft_budget_param": None,
            "mean_accuracy": 0.0,
            "std_accuracy": None,
            "mean_f_score": 0.0,
            "std_f_score": None,
            "mean_avg_accumulated_cost": 1.0,
            "std_avg_accumulated_cost": None,
            "mean_metric": 0.0,
            "std_metric": None,
            "low_metric": None,
            "high_metric": None,
            "low_avg_accumulated_cost": None,
            "high_avg_accumulated_cost": None,
        },
        {
            "afa_method": "jafa",
            "dataset": "cube",
            "eval_hard_budget": 2.0,
            "soft_budget_param": None,
            # Seeds score 0.5 and 1.0
            "mean_accuracy": 0.75,
            "std_accuracy": pytest.approx(0.5 * SQRT_HALF),
            # Seeds score 1/3 and 1.0
            "mean_f_score": pytest.approx(2 / 3),
            "std_f_score": pytest.approx(2 / 3 * SQRT_HALF),
            # Seeds average 1.5 and 2.0
            "mean_avg_accumulated_cost": 1.75,
            "std_avg_accumulated_cost": pytest.approx(0.5 * SQRT_HALF),
            # cube is scored with accuracy
            "mean_metric": 0.75,
            "std_metric": pytest.approx(0.5 * SQRT_HALF),
            "low_metric": pytest.approx(0.75 - 0.5 * SQRT_HALF),
            "high_metric": pytest.approx(0.75 + 0.5 * SQRT_HALF),
            "low_avg_accumulated_cost": pytest.approx(1.75 - 0.5 * SQRT_HALF),
            "high_avg_accumulated_cost": pytest.approx(1.75 + 0.5 * SQRT_HALF),
        },
    ]


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_per_time_step_metrics_use_largest_hard_budget_per_dataset(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    plotter = EvaluationPlotter(eval_perf_path, tmp_path, plotting_config)

    plotter.load_and_process()

    assert plotter.df_traj is not None
    # Only JAFA on cube at hard budget 2 remains: budget 1 is not the
    # largest, and the soft budget run has no hard budget at all
    shared = {
        "afa_method": "jafa",
        "dataset": "cube",
        "eval_hard_budget": 2.0,
        "soft_budget_param": None,
    }
    wrong_once = {
        "mean_accuracy": 0.0,
        "std_accuracy": None,
        "mean_f_score": 0.0,
        "std_f_score": None,
        "mean_metric": 0.0,
        "std_metric": None,
        "low_metric": None,
        "high_metric": None,
    }
    assert sorted_rows(plotter.df_traj, "n_selections_performed") == [
        shared | {"n_selections_performed": 0} | wrong_once,
        shared | {"n_selections_performed": 1} | wrong_once,
        shared
        | {
            "n_selections_performed": 2,
            "mean_accuracy": 1.0,
            "std_accuracy": 0.0,
            "mean_f_score": 1.0,
            "std_f_score": 0.0,
            "mean_metric": 1.0,
            "std_metric": 0.0,
            "low_metric": 1.0,
            "high_metric": 1.0,
        },
    ]


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_all_plots_are_written_for_each_dataset_set(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    output_folder = tmp_path / "plots"
    plotter = EvaluationPlotter(
        eval_perf_path, output_folder, plotting_config, formats=("png",)
    )

    plotter.load_and_process()
    plotter.generate_all_plots()

    assert sorted(path.name for path in (output_folder / "all").iterdir()) == [
        "hard_budget_normal.png",
        "hard_budget_traj.png",
        "soft_budget_2d_errors.png",
        "soft_budget_lines.png",
    ]


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_caption_labels_every_plot(
    eval_perf_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    output_folder = tmp_path / "plots"
    caption = "Workflow demonstration, not scientific results"
    plotter = EvaluationPlotter(
        eval_perf_path,
        output_folder,
        replace(plotting_config, caption=caption),
        formats=("svg",),
    )

    plotter.load_and_process()
    plotter.generate_all_plots()

    plots = sorted((output_folder / "all").iterdir())
    assert len(plots) == 4
    # Matplotlib keeps each text of an SVG as a comment beside its glyphs.
    for plot in plots:
        assert f"<!-- {caption} -->" in plot.read_text(), plot.name
