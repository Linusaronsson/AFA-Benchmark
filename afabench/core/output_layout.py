"""
The pipeline's native output layout of bundles, tables and job records.

Every Snakemake rule addresses bundles, evaluation tables and the job
records beside them (`docs/reference/job_records.md`) through this module,
so their layout is spelled once; the aggregation and visualization rules
spell their merged results and plots themselves, under this module's
initializer tag. Benchmark releases are restored at these paths and
reference methods' tables are found by them, so the layout must not change.
Artifacts carry their own identity
(`docs/adr/0002-provenance-recorded-in-artifacts.md`); the paths are
Snakemake target naming only.

The same builders produce rule patterns and concrete targets: pass
`"{train_seed}"`-style placeholders for wildcards and values for targets.
Snakemake imports this module at parse time, so it must not import torch.
"""

import re
from dataclasses import dataclass
from typing import Self

type PathValue = str | int | float

_PRETRAIN_SEED_PREFIX = "pretrain_seed-"
_NO_PRETRAIN_FOLDER = "NO_PRETRAIN"
# Constrains a `{pretrain_folder}` wildcard to the folders below.
PRETRAIN_FOLDER_PATTERN = rf"{_PRETRAIN_SEED_PREFIX}\d+|{_NO_PRETRAIN_FOLDER}"


def pretrain_seed_folder(pretrain_seed: PathValue | None) -> str:
    """Name the folder of one pretraining seed; `None` for no such stage."""
    if pretrain_seed is None:
        return _NO_PRETRAIN_FOLDER
    return f"{_PRETRAIN_SEED_PREFIX}{pretrain_seed}"


def pretrain_seed_in_folder(folder: str) -> str | None:
    """Read back the seed `pretrain_seed_folder` wrote into `folder`."""
    if not re.fullmatch(PRETRAIN_FOLDER_PATTERN, folder):
        msg = f"Not a pretrain folder: {folder!r}"
        raise ValueError(msg)
    if folder == _NO_PRETRAIN_FOLDER:
        return None
    return folder.removeprefix(_PRETRAIN_SEED_PREFIX)


@dataclass(frozen=True, kw_only=True)
class TrainingRun:
    """
    The path segments of one trained method.

    `pretrain_folder` comes from `pretrain_seed_folder`, or is a placeholder
    in a rule pattern that covers methods with and without a pretraining
    stage.
    """

    method: PathValue
    dataset: PathValue
    dataset_realization_index: PathValue
    pretrain_folder: str
    train_seed: PathValue
    train_hard_budget: PathValue
    train_soft_budget_param: PathValue

    @classmethod
    def wildcards(cls, *, pretrain_folder: str = "{pretrain_folder}") -> Self:
        """Build the rule pattern: each field is a wildcard of its name."""
        return cls(
            method="{method}",
            dataset="{dataset}",
            dataset_realization_index="{dataset_realization_index}",
            pretrain_folder=pretrain_folder,
            train_seed="{train_seed}",
            train_hard_budget="{train_hard_budget}",
            train_soft_budget_param="{train_soft_budget_param}",
        )


@dataclass(frozen=True, kw_only=True)
class EvaluationRun:
    """The path segments of one evaluation of a trained method."""

    training: TrainingRun
    eval_seed: PathValue
    eval_hard_budget: PathValue
    eval_soft_budget_param: PathValue

    @classmethod
    def wildcards(cls, *, pretrain_folder: str = "{pretrain_folder}") -> Self:
        """Build the rule pattern: each field is a wildcard of its name."""
        return cls(
            training=TrainingRun.wildcards(pretrain_folder=pretrain_folder),
            eval_seed="{eval_seed}",
            eval_hard_budget="{eval_hard_budget}",
            eval_soft_budget_param="{eval_soft_budget_param}",
        )


@dataclass(frozen=True, kw_only=True)
class OutputLayout:
    """Paths under `root` for one initializer and evaluation split."""

    root: str
    initializer: str
    eval_split: str

    def dataset_folder(self, *, dataset: PathValue) -> str:
        """Where dataset generation writes every realization's bundles."""
        return self._path("datasets", f"{dataset}")

    def dataset_generation_job_record(self, *, dataset: PathValue) -> str:
        return "/".join(
            [
                self.dataset_folder(dataset=dataset),
                "dataset_generation.job_record.json",
            ]
        )

    def dataset_bundle(
        self,
        *,
        dataset: PathValue,
        dataset_realization_index: PathValue,
        split: PathValue,
    ) -> str:
        return "/".join(
            [
                self.dataset_folder(dataset=dataset),
                f"{dataset_realization_index}",
                f"{split}.bundle",
            ]
        )

    def classifier_bundle(
        self,
        *,
        dataset: PathValue,
        dataset_realization_index: PathValue,
        method: PathValue | None,
    ) -> str:
        """Address `method`'s built-in classifier; None for the external."""
        return self._classifier_path(
            dataset, dataset_realization_index, method, ".bundle"
        )

    def classifier_job_record(
        self,
        *,
        dataset: PathValue,
        dataset_realization_index: PathValue,
        method: PathValue | None,
    ) -> str:
        return self._classifier_path(
            dataset, dataset_realization_index, method, ".job_record.json"
        )

    def _classifier_path(
        self,
        dataset: PathValue,
        dataset_realization_index: PathValue,
        method: PathValue | None,
        suffix: str,
    ) -> str:
        realization = _dataset_realization_folder(
            dataset, dataset_realization_index
        )
        name = (
            realization if method is None else f"method-{method}+{realization}"
        )
        return self._path(
            "trained_classifiers", self.initializer_tag, f"{name}{suffix}"
        )

    def pretrained_model_bundle(
        self,
        *,
        pretrained_model_name: PathValue,
        dataset: PathValue,
        dataset_realization_index: PathValue,
        pretrain_seed: PathValue,
    ) -> str:
        return self._pretraining_path(
            pretrained_model_name,
            dataset,
            dataset_realization_index,
            pretrain_seed,
            "model.bundle",
        )

    def pretraining_job_record(
        self,
        *,
        pretrained_model_name: PathValue,
        dataset: PathValue,
        dataset_realization_index: PathValue,
        pretrain_seed: PathValue,
    ) -> str:
        return self._pretraining_path(
            pretrained_model_name,
            dataset,
            dataset_realization_index,
            pretrain_seed,
            "model.job_record.json",
        )

    def method_bundle(self, run: TrainingRun) -> str:
        return self._training_path(run, "method.bundle")

    def training_job_record(self, run: TrainingRun) -> str:
        return self._training_path(run, "method.job_record.json")

    def _training_path(self, run: TrainingRun, file_name: str) -> str:
        return self._path(
            "trained_methods",
            self.initializer_tag,
            *_training_segments(run),
            file_name,
        )

    def raw_evaluation_table(self, run: EvaluationRun) -> str:
        return self._evaluation_path("eval_results", run, "eval_data.parquet")

    def transformed_evaluation_table(self, run: EvaluationRun) -> str:
        return self._evaluation_path(
            "eval_results_transformed", run, "eval_data.parquet"
        )

    def evaluation_job_record(self, run: EvaluationRun) -> str:
        return self._evaluation_path(
            "eval_results", run, "eval_data.job_record.json"
        )

    def transformation_job_record(self, run: EvaluationRun) -> str:
        return self._evaluation_path(
            "eval_results_transformed", run, "eval_data.job_record.json"
        )

    def _evaluation_path(
        self, stage_folder: str, run: EvaluationRun, file_name: str
    ) -> str:
        return self._path(
            stage_folder,
            f"eval_split-{self.eval_split}",
            self.initializer_tag,
            *_training_segments(run.training),
            f"eval_seed-{run.eval_seed}+"
            f"eval_hard_budget-{run.eval_hard_budget}+"
            f"eval_soft_budget_param-{run.eval_soft_budget_param}",
            file_name,
        )

    def _pretraining_path(
        self,
        pretrained_model_name: PathValue,
        dataset: PathValue,
        dataset_realization_index: PathValue,
        pretrain_seed: PathValue,
        file_name: str,
    ) -> str:
        return self._path(
            "pretrained_models",
            self.initializer_tag,
            f"{pretrained_model_name}",
            _dataset_realization_folder(dataset, dataset_realization_index),
            pretrain_seed_folder(pretrain_seed),
            file_name,
        )

    @property
    def initializer_tag(self) -> str:
        return f"initializer-{self.initializer}"

    def _path(self, *segments: str) -> str:
        return "/".join([self.root, *segments])


def _dataset_realization_folder(
    dataset: PathValue, dataset_realization_index: PathValue
) -> str:
    return f"dataset-{dataset}+realization_index-{dataset_realization_index}"


def _training_segments(run: TrainingRun) -> list[str]:
    return [
        f"{run.method}",
        _dataset_realization_folder(
            run.dataset, run.dataset_realization_index
        ),
        run.pretrain_folder,
        f"train_seed-{run.train_seed}+"
        f"train_hard_budget-{run.train_hard_budget}+"
        f"train_soft_budget_param-{run.train_soft_budget_param}",
    ]
