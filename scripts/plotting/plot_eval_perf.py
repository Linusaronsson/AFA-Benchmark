from pathlib import Path
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.evaluation.tables import load_plotting_ready_table
from afabench.plotting.config import PlotEvalPerfConfig
from afabench.plotting.eval_perf import EvaluationPlotter


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/plotting/plot_eval_perf",
    config_name="config",
)
def main(cfg: PlotEvalPerfConfig) -> None:
    cfg = cast("PlotEvalPerfConfig", OmegaConf.to_object(cfg))
    assert isinstance(cfg, PlotEvalPerfConfig)

    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    plotter = EvaluationPlotter(
        load_plotting_ready_table(Path(cfg.input)),
        output_folder,
        cfg.plotting,
        cfg.formats,
    )
    plotter.generate_all_plots(cfg.dataset_sets)


if __name__ == "__main__":
    main()
