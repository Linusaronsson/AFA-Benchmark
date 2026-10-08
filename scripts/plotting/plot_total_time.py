"""
Plot the time each method's jobs take per pipeline stage, from the job duration table.

The input is a job duration table (`docs/reference/job_records.md`). Only
completed jobs count, so failed and timed-out attempts are left out. A job
belongs to a method if its record sits in that method's folders of the
selected initializer and evaluation split: its pretraining (through the
pretrained model the method trains from), its method-specific classifier
training, its training, its evaluation and its transformation. Dataset
generation and shared classifiers belong to no method.

The averaged plot shows each stage's mean job duration over the datasets
every method has jobs on; the per-dataset plot stacks every job, so it
shows each stage's total.
"""

from pathlib import Path
from typing import cast

import hydra
import pandas as pd
import plotnine as p9
from omegaconf import OmegaConf
from plotnine import (
    aes,
    coord_flip,
    element_text,
    facet_wrap,
    geom_bar,
    ggplot,
    labs,
    scale_fill_brewer,
    scale_x_discrete,
    theme,
)

from afabench.core.job_duration_table import load_job_duration_table
from afabench.plotting.config import PlottingDisplayConfig, PlotTotalTimeConfig

PLOT_FONT_SIZE = 12

# Pipeline stages a method's jobs run in, in pipeline order
STAGE_LABELS = {
    "classifier_training": "Classifier training",
    "pretraining": "Pretraining",
    "training": "Training",
    "evaluation": "Evaluation",
    "transformation": "Transformation",
}


def method_job_durations(
    table: pd.DataFrame,
    *,
    methods: list[str],
    pretrained_models: dict[str, str],
    initializer_tag: str,
    eval_dataset_split: str,
) -> pd.DataFrame:
    """
    Return the job duration of each completed job of each method.

    `pretrained_models` maps a method to the pretrained model it trains
    from. Columns: afa_method, dataset, stage, time (seconds); a pretraining
    job shared by several methods is a row of each.
    """
    completed = table.loc[table["exit_status"].eq("completed").fillna(False)]
    split_tag = f"eval_split-{eval_dataset_split}/{initializer_tag}"
    method_jobs = []
    for method in methods:
        folders = {
            "classifier_training": f"trained_classifiers/{initializer_tag}/method-{method}+",
            "training": f"trained_methods/{initializer_tag}/{method}/",
            "evaluation": f"eval_results/{split_tag}/{method}/",
            "transformation": f"eval_results_transformed/{split_tag}/{method}/",
        }
        if method in pretrained_models:
            folders["pretraining"] = (
                f"pretrained_models/{initializer_tag}/"
                f"{pretrained_models[method]}/"
            )
        for stage, folder in folders.items():
            jobs = completed.loc[
                completed["stage"].eq(stage).fillna(False)
                & completed["job_record_path"].str.startswith(folder)
            ]
            method_jobs.append(
                pd.DataFrame(
                    {
                        "afa_method": method,
                        "dataset": jobs["dataset_key"],
                        "stage": stage,
                        "time": jobs["job_duration_seconds"],
                    }
                )
            )
    return pd.concat(method_jobs, ignore_index=True).astype(
        {
            "afa_method": "string",
            "dataset": "string",
            "stage": "string",
            "time": "float64",
        }
    )


def common_plot_operations(
    p: p9.ggplot, plotting_config: PlottingDisplayConfig
) -> p9.ggplot:
    return (
        p
        + coord_flip()
        + labs(x="Policy", y="Time (s)", fill="Stage")
        + theme(
            text=element_text(
                family=plotting_config.plot_font_family,
                size=PLOT_FONT_SIZE,
            )
        )
        + scale_x_discrete()
        + scale_fill_brewer(
            type="qual",
            palette=plotting_config.color_palette_name,
            labels=STAGE_LABELS,
            breaks=list(STAGE_LABELS),
        )
    )


def filter_common_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter dataframe to only include datasets present in all methods.

    This ensures that the averaging plot is fair - methods that only train on
    a subset of datasets won't be penalized by missing data from larger datasets.
    """
    # Get the set of datasets for each method
    method_dataset_sets = [
        set(datasets) for _, datasets in df.groupby("afa_method")["dataset"]
    ]

    # Find datasets that appear in all methods
    common_datasets = set.intersection(*method_dataset_sets)

    # Filter the dataframe to only include common datasets
    return df.loc[df["dataset"].isin(list(common_datasets))]


def get_plots(
    df: pd.DataFrame, plotting_config: PlottingDisplayConfig
) -> tuple[p9.ggplot, p9.ggplot]:
    # Apply name transforms
    df = df.assign(
        dataset=df["dataset"].replace(plotting_config.dataset_name_mapping),
        afa_method=df["afa_method"].replace(
            plotting_config.method_name_mapping
        ),
    )

    # For averaging plot, only use datasets present in all methods
    df_common = filter_common_datasets(df)

    # Get display method order based on METHOD_NAME_MAPPING order.
    # After name transformation, values are already display names.
    method_order = [
        plotting_config.method_name_mapping.get(m, m)
        for m in plotting_config.method_name_mapping
    ]
    # Reverse the order to match the desired display.
    method_order.reverse()

    # Filter to only mapped methods present in the data.
    available_methods = set(df["afa_method"].unique().tolist())
    method_order_filtered = [m for m in method_order if m in available_methods]
    # Include methods that are present in data but not in METHOD_NAME_MAPPING.
    # This keeps new/experimental methods from becoming nulls in the cast.
    unknown_methods = sorted(available_methods - set(method_order_filtered))
    method_order_filtered.extend(unknown_methods)

    # Cast to categorical with the correct order. Stages stack in reverse
    # pipeline order, the last stage at the bottom.
    dtypes = {
        "afa_method": pd.CategoricalDtype(method_order_filtered),
        "stage": pd.CategoricalDtype(list(reversed(STAGE_LABELS))),
    }
    df_common = df_common.astype(dtypes)
    df = df.astype(dtypes)

    # One plot averaged over datasets (using only common datasets)
    averaged_plot = ggplot(
        df_common.groupby(["afa_method", "stage"], observed=True)["time"]
        .mean()
        .reset_index()
    ) + geom_bar(
        aes(x="afa_method", y="time", fill="stage"),
        stat="identity",
    )
    averaged_plot = common_plot_operations(averaged_plot, plotting_config)

    # Another one faceted over datasets (showing all datasets)
    dataset_plot = (
        ggplot(df)
        + geom_bar(
            aes(x="afa_method", y="time", fill="stage"), stat="identity"
        )
        + facet_wrap("dataset", nrow=2, scales="free_x")
    )
    dataset_plot = common_plot_operations(dataset_plot, plotting_config)
    return averaged_plot, dataset_plot


@hydra.main(
    version_base=None,
    config_path="../../extra/conf/scripts/plotting/plot_total_time",
    config_name="config",
)
def main(cfg: PlotTotalTimeConfig) -> None:
    cfg = cast("PlotTotalTimeConfig", OmegaConf.to_object(cfg))
    assert isinstance(cfg, PlotTotalTimeConfig)

    durations = method_job_durations(
        load_job_duration_table(Path(cfg.input)),
        methods=cfg.methods,
        pretrained_models=cfg.pretrained_models,
        initializer_tag=cfg.initializer_tag,
        eval_dataset_split=cfg.eval_dataset_split,
    )

    averaged_plot, dataset_plot = get_plots(
        df=durations, plotting_config=cfg.plotting
    )

    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    for fmt in cfg.formats:
        averaged_plot.save(
            output_folder / f"average_time.{fmt}",
            width=10,
            height=3,
            verbose=False,
        )
        dataset_plot.save(
            output_folder / f"dataset_time.{fmt}",
            width=20,
            height=5,
            verbose=False,
        )


if __name__ == "__main__":
    main()
