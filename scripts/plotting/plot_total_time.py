"""
Plot the time each method's jobs take per pipeline stage, from the job duration table.

What counts as a method's job: `afabench.plotting.total_time`.
"""

from pathlib import Path
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.core.job_duration_table import load_job_duration_table
from afabench.plotting.config import PlotTotalTimeConfig
from afabench.plotting.total_time import method_job_durations, save_time_plots


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/plotting/plot_total_time",
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
    save_time_plots(
        durations, Path(cfg.output_folder), cfg.plotting, cfg.formats
    )


if __name__ == "__main__":
    main()
