from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


@dataclass
class TrainingTimeCalculationConfig:
    plotting_run_names: list[str]
    output_artifact_aliases: list[str]
    max_workers: int


cs.store(name="training_time_calculation", node=TrainingTimeCalculationConfig)


@dataclass
class EvaluationTimeCalculationConfig:
    plotting_run_names: list[str]
    output_artifact_aliases: list[str]
    max_workers: int


cs.store(
    name="evaluation_time_calculation", node=EvaluationTimeCalculationConfig
)


@dataclass
class PlotDownloadConfig:
    plotting_run_name: str
    datasets: list[str]
    metrics: list[str]
    budgets: list[str]
    file_type: str
    output_path: str
