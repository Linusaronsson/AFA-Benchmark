from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


@dataclass
class MetricConfig:
    key: str
    description: str
    ylim: list[int] | None


@dataclass
class PlottingDisplayConfig:
    plot_width: float
    plot_height: float
    plot_font_family: str
    method_name_mapping: dict[str, str]
    method_policy_family_mapping: dict[str, str]
    method_family_color_schemes: dict[str, dict[str, str]]
    active_method_color_scheme: str
    dataset_name_mapping: dict[str, str]
    datasets_with_f_score: list[str]
    dataset_sets: dict[str, list[str]]
    color_palette_name: str


@dataclass
class PlotConfig:
    eval_artifact_yaml_list: str
    metric_keys_and_descriptions: list[MetricConfig]
    max_workers: int


cs.store(name="plot", node=PlotConfig)


@dataclass
class PlotEvalPerfConfig:
    input: str
    output_folder: str
    formats: list[str]
    plotting: PlottingDisplayConfig


cs.store(name="plot_eval_perf", node=PlotEvalPerfConfig)


@dataclass
class PlotEvalActionsConfig:
    input: str
    output_folder: str
    formats: list[str]
    plotting: PlottingDisplayConfig


cs.store(name="plot_eval_actions", node=PlotEvalActionsConfig)


@dataclass
class PlotTotalTimeConfig:
    output_folder: str
    input: str | None
    formats: list[str]
    plotting: PlottingDisplayConfig


cs.store(name="plot_total_time", node=PlotTotalTimeConfig)


@dataclass
class PlotMissingDataConfig:
    instance_metrics: str
    summary: str
    action_rates: str
    restoration_rmse: str
    output_folder: str
    formats: list[str]
    plotting: PlottingDisplayConfig


cs.store(name="plot_missing_data", node=PlotMissingDataConfig)
