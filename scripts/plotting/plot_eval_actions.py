from pathlib import Path
from typing import cast

import hydra
from omegaconf import OmegaConf

from afabench.evaluation.tables import load_plotting_ready_table
from afabench.plotting.config import PlotEvalActionsConfig
from afabench.plotting.eval_actions import save_all_action_heatmaps


@hydra.main(
    version_base=None,
    config_path="../../conf/scripts/plotting/plot_eval_actions",
    config_name="config",
)
def main(cfg: PlotEvalActionsConfig) -> None:
    cfg = cast("PlotEvalActionsConfig", OmegaConf.to_object(cfg))
    assert isinstance(cfg, PlotEvalActionsConfig)

    save_all_action_heatmaps(
        load_plotting_ready_table(Path(cfg.input)),
        Path(cfg.output_folder),
        cfg.plotting,
        cfg.formats,
    )


if __name__ == "__main__":
    main()
