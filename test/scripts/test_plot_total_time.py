from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from afabench.plotting.config import (
    PlottingDisplayConfig,
    PlotTotalTimeConfig,
)
from scripts.plotting.plot_total_time import (
    get_plots,
    main,
    read_parquet_safe,
    unpivot,
)


@pytest.fixture
def time_results_path(tmp_path: Path) -> Path:
    path = tmp_path / "time.parquet"
    columns = [
        "afa_method",
        "dataset",
        "time_pretrain",
        "time_train",
        "time_eval",
    ]
    rows = [
        ("jafa", "cube", 1.0, 10.0, 2.0),
        ("jafa", "cube", 3.0, 20.0, 4.0),
        ("jafa", "mnist", None, 100.0, 50.0),
        ("gdfs", "cube", None, 5.0, 1.0),
        # Not in the method name mapping
        ("other", "cube", 2.0, 2.0, 2.0),
    ]
    pq.write_table(
        pa.Table.from_pylist(
            [dict(zip(columns, row, strict=True)) for row in rows],
            schema=pa.schema(
                {
                    "afa_method": pa.string(),
                    "dataset": pa.string(),
                    "time_pretrain": pa.float64(),
                    "time_train": pa.float64(),
                    "time_eval": pa.float64(),
                }
            ),
        ),
        path,
    )
    return path


def category_order(df: object, column: str) -> list[str]:
    return pa.table(df)[column].combine_chunks().dictionary.to_pylist()


def test_average_uses_datasets_shared_by_all_methods(
    time_results_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    df_long = unpivot(read_parquet_safe(time_results_path))

    averaged_plot, _ = get_plots(df_long, plotting_config)

    rows = pa.table(averaged_plot.data).select(["afa_method", "stage", "time"])
    # Only cube is shared by all methods; missing pretraining counts as 0
    assert sorted(
        rows.to_pylist(), key=lambda r: (r["afa_method"], r["stage"])
    ) == [
        {"afa_method": "GDFS", "stage": "eval", "time": 1.0},
        {"afa_method": "GDFS", "stage": "pretrain", "time": 0.0},
        {"afa_method": "GDFS", "stage": "train", "time": 5.0},
        {"afa_method": "JAFA", "stage": "eval", "time": 3.0},
        {"afa_method": "JAFA", "stage": "pretrain", "time": 2.0},
        {"afa_method": "JAFA", "stage": "train", "time": 15.0},
        {"afa_method": "other", "stage": "eval", "time": 2.0},
        {"afa_method": "other", "stage": "pretrain", "time": 2.0},
        {"afa_method": "other", "stage": "train", "time": 2.0},
    ]


def test_methods_follow_reversed_mapping_order_and_stages_stack_eval_first(
    time_results_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    df_long = unpivot(read_parquet_safe(time_results_path))

    averaged_plot, dataset_plot = get_plots(df_long, plotting_config)

    for plot in (averaged_plot, dataset_plot):
        assert category_order(plot.data, "afa_method") == [
            "GDFS",
            "JAFA",
            "other",
        ]
        assert category_order(plot.data, "stage") == [
            "eval",
            "train",
            "pretrain",
        ]
    # The per-dataset plot keeps every row, with display names applied
    assert sorted(set(pa.table(dataset_plot.data)["dataset"].to_pylist())) == [
        "Cube",
        "MNIST",
    ]
    assert pa.table(dataset_plot.data).num_rows == 15


def test_main_writes_average_and_per_dataset_plots(
    time_results_path: Path,
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    output_folder = tmp_path / "plots"
    cfg = OmegaConf.structured(
        PlotTotalTimeConfig(
            input=str(time_results_path),
            output_folder=str(output_folder),
            formats=["png"],
            plotting=plotting_config,
        )
    )

    main(cfg)

    assert sorted(path.name for path in output_folder.iterdir()) == [
        "average_time.png",
        "dataset_time.png",
    ]
