from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from omegaconf import OmegaConf

from afabench.core.job_duration_table import COLUMN_DTYPES
from afabench.plotting.config import (
    PlottingDisplayConfig,
    PlotTotalTimeConfig,
)
from scripts.plotting.plot_total_time import (
    get_plots,
    main,
    method_job_durations,
)

TAG = "initializer-cold"
EVALUATION = "eval_seed-0+eval_hard_budget-1+eval_soft_budget_param-null"


def training(
    method: str, dataset: str = "cube", pretrain: str = "NO_PRETRAIN"
) -> str:
    return (
        f"{method}/dataset-{dataset}+realization_index-0/{pretrain}/"
        "train_seed-0+train_hard_budget-1+train_soft_budget_param-null"
    )


def training_record(method: str, **kwargs: str) -> str:
    return f"trained_methods/{TAG}/{training(method, **kwargs)}/method.job_record.json"


def evaluation_record(
    method: str,
    stage: str = "eval_results",
    split: str = "test",
    **kwargs: str,
) -> str:
    return (
        f"{stage}/eval_split-{split}/{TAG}/{training(method, **kwargs)}/"
        f"{EVALUATION}/eval_data.job_record.json"
    )


def job_duration_table(
    rows: list[tuple[str, str, str, float, str]],
) -> pd.DataFrame:
    """Rows of (job_record_path, stage, dataset_key, duration, exit_status)."""
    table = pd.DataFrame(
        [
            {
                "job_record_path": path,
                "job_record_version": 1,
                "stage": stage,
                "dataset_key": dataset,
                "job_duration_seconds": duration,
                "exit_status": exit_status,
            }
            for path, stage, dataset, duration, exit_status in rows
        ],
        columns=pd.Index(COLUMN_DTYPES),
    )
    return table.astype(COLUMN_DTYPES)


def completed(
    path: str, stage: str, duration: float, dataset: str = "cube"
) -> tuple[str, str, str, float, str]:
    return (path, stage, dataset, duration, "completed")


def test_each_selected_method_gets_its_completed_jobs_of_this_run() -> None:
    jafa_pretrain = (
        f"pretrained_models/{TAG}/shared/dataset-cube+realization_index-0/"
        "pretrain_seed-0/model.job_record.json"
    )
    table = job_duration_table(
        [
            # Not a method's: dataset generation and the shared classifier
            completed(
                "datasets/cube.job_record.json", "dataset_generation", 100
            ),
            completed(
                f"trained_classifiers/{TAG}/dataset-cube+realization_index-0.job_record.json",
                "classifier_training",
                50,
            ),
            completed(
                f"trained_classifiers/{TAG}/method-jafa+dataset-cube+realization_index-0.job_record.json",
                "classifier_training",
                7,
            ),
            completed(jafa_pretrain, "pretraining", 3),
            completed(
                training_record("jafa", pretrain="pretrain_seed-0"),
                "training",
                10,
            ),
            completed(
                evaluation_record("jafa", pretrain="pretrain_seed-0"),
                "evaluation",
                2,
            ),
            completed(
                evaluation_record(
                    "jafa",
                    stage="eval_results_transformed",
                    pretrain="pretrain_seed-0",
                ),
                "transformation",
                1,
            ),
            # Failed and timed-out attempts
            (
                "failed_job_records/"
                + training_record("jafa", pretrain="pretrain_seed-0").replace(
                    "method.job_record.json",
                    "method.20261008T120000123456Z-1a2b3c4d.job_record.json",
                ),
                "training",
                "cube",
                1000,
                "failed",
            ),
            (
                evaluation_record("gdfs"),
                "evaluation",
                "cube",
                2000,
                "timeout",
            ),
            completed(training_record("gdfs"), "training", 5),
            completed(evaluation_record("gdfs"), "evaluation", 1),
            # Another initializer, evaluation split or method
            completed(
                training_record("gdfs").replace(TAG, "initializer-warm"),
                "training",
                500,
            ),
            completed(
                evaluation_record("gdfs", split="val"), "evaluation", 600
            ),
            completed(training_record("odin"), "training", 900),
        ]
    )

    durations = method_job_durations(
        table,
        methods=["gdfs", "jafa"],
        pretrained_models={"jafa": "shared"},
        initializer_tag=TAG,
        eval_dataset_split="test",
    )

    assert sorted(durations.itertuples(index=False, name=None)) == [
        ("gdfs", "cube", "evaluation", 1.0),
        ("gdfs", "cube", "training", 5.0),
        ("jafa", "cube", "classifier_training", 7.0),
        ("jafa", "cube", "evaluation", 2.0),
        ("jafa", "cube", "pretraining", 3.0),
        ("jafa", "cube", "training", 10.0),
        ("jafa", "cube", "transformation", 1.0),
    ]
    assert list(durations.columns) == [
        "afa_method",
        "dataset",
        "stage",
        "time",
    ]


def test_a_pretrained_model_counts_for_every_method_trained_from_it() -> None:
    table = job_duration_table(
        [
            completed(
                f"pretrained_models/{TAG}/shared/dataset-cube+realization_index-0/"
                "pretrain_seed-0/model.job_record.json",
                "pretraining",
                3,
            ),
        ]
    )

    durations = method_job_durations(
        table,
        methods=["jafa", "odin"],
        pretrained_models={"jafa": "shared", "odin": "shared"},
        initializer_tag=TAG,
        eval_dataset_split="test",
    )

    assert sorted(durations.itertuples(index=False, name=None)) == [
        ("jafa", "cube", "pretraining", 3.0),
        ("odin", "cube", "pretraining", 3.0),
    ]


@pytest.fixture
def durations() -> pd.DataFrame:
    rows = [
        ("jafa", "cube", "pretraining", 2.0),
        ("jafa", "cube", "training", 10.0),
        ("jafa", "cube", "training", 20.0),
        ("jafa", "cube", "evaluation", 3.0),
        ("jafa", "mnist", "training", 100.0),
        ("jafa", "mnist", "evaluation", 50.0),
        ("gdfs", "cube", "training", 5.0),
        ("gdfs", "cube", "evaluation", 1.0),
        ("gdfs", "cube", "transformation", 0.5),
        # Not in the method name mapping
        ("other", "cube", "classifier_training", 2.0),
        ("other", "cube", "training", 2.0),
    ]
    return pd.DataFrame(
        rows, columns=pd.Index(["afa_method", "dataset", "stage", "time"])
    )


def category_order(df: object, column: str) -> list[str]:
    return pa.table(df)[column].combine_chunks().dictionary.to_pylist()


def test_average_is_the_mean_job_duration_over_datasets_shared_by_all_methods(
    durations: pd.DataFrame,
    plotting_config: PlottingDisplayConfig,
) -> None:
    averaged_plot, _ = get_plots(durations, plotting_config)

    rows = pa.table(averaged_plot.data).select(["afa_method", "stage", "time"])
    # Only cube is shared by all methods
    assert sorted(
        rows.to_pylist(), key=lambda r: (r["afa_method"], r["stage"])
    ) == [
        {"afa_method": "GDFS", "stage": "evaluation", "time": 1.0},
        {"afa_method": "GDFS", "stage": "training", "time": 5.0},
        {"afa_method": "GDFS", "stage": "transformation", "time": 0.5},
        {"afa_method": "JAFA", "stage": "evaluation", "time": 3.0},
        {"afa_method": "JAFA", "stage": "pretraining", "time": 2.0},
        {"afa_method": "JAFA", "stage": "training", "time": 15.0},
        {"afa_method": "other", "stage": "classifier_training", "time": 2.0},
        {"afa_method": "other", "stage": "training", "time": 2.0},
    ]


def test_methods_follow_reversed_mapping_order_and_stages_stack_last_first(
    durations: pd.DataFrame,
    plotting_config: PlottingDisplayConfig,
) -> None:
    averaged_plot, dataset_plot = get_plots(durations, plotting_config)

    for plot in (averaged_plot, dataset_plot):
        assert category_order(plot.data, "afa_method") == [
            "GDFS",
            "JAFA",
            "other",
        ]
        assert category_order(plot.data, "stage") == [
            "transformation",
            "evaluation",
            "training",
            "pretraining",
            "classifier_training",
        ]
    # The per-dataset plot keeps every job, with display names applied
    assert sorted(set(pa.table(dataset_plot.data)["dataset"].to_pylist())) == [
        "Cube",
        "MNIST",
    ]
    assert pa.table(dataset_plot.data).num_rows == len(durations)


def test_main_plots_a_job_duration_table(
    tmp_path: Path,
    plotting_config: PlottingDisplayConfig,
) -> None:
    table_path = tmp_path / "job_duration_table.parquet"
    job_duration_table(
        [
            completed(training_record("gdfs"), "training", 5),
            completed(evaluation_record("gdfs"), "evaluation", 1),
        ]
    ).to_parquet(table_path, index=False)
    output_folder = tmp_path / "plots"
    cfg = OmegaConf.structured(
        PlotTotalTimeConfig(
            input=str(table_path),
            output_folder=str(output_folder),
            methods=["gdfs"],
            pretrained_models={},
            initializer_tag=TAG,
            eval_dataset_split="test",
            formats=["png"],
            plotting=plotting_config,
        )
    )

    main(cfg)

    assert sorted(path.name for path in output_folder.iterdir()) == [
        "average_time.png",
        "dataset_time.png",
    ]
