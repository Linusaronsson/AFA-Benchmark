"""
The release manifest: identity, provenance and coverage of a snapshot.

A release manifest is a JSON file written beside an output snapshot's
`output/` tree. It is filled from the workflow configuration and the git
state of the checkout. Per-table identity is enumerated forward from the
resolved workflow configuration, the same way the workflow's
`all_eval_methods` rule names its targets, rather than parsed back out of
paths. Once evaluation tables carry their own identity columns and
provenance record (`docs/adr/0002-provenance-recorded-in-artifacts.md`),
that enumeration should read them instead. The field list is documented in
`docs/release_manifest.md`.
"""

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import dacite
import pyarrow.parquet as pq

from afabench.core.types import FEATURE_COSTS_DIR
from afabench.release.workflow_config import (
    WorkflowConfigRecord,
    load_workflow_settings,
)

MANIFEST_VERSION = 1
RELEASE_MANIFEST_FILENAME = "release_manifest.json"
# The dataset generation rule always writes these three split bundles.
DATASET_SPLITS = ("train", "val", "test")
# Pretrained models, trained methods and evaluations of a dataset instance
# are all seeded with its index, as are the dataset generators. Every
# classifier is trained on instance 0 with seed 0.
CLASSIFIER_DATASET_INSTANCE_INDEX = 0
CLASSIFIER_SEED = 0
FORCING_POLICY = "forced_acquisition_when_eval_hard_budget_is_set"


class ReleaseScope(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    TEST_ONLY = "test_only"


class ExecutionMode(StrEnum):
    SMOKE = "smoke"
    PRODUCTION = "production"


class BudgetSetting(StrEnum):
    HARD_BUDGET = "hard_budget"
    SOFT_BUDGET = "soft_budget"


class ClassifierVariant(StrEnum):
    """Prediction columns of a raw table; `classifier` after transform."""

    BUILTIN = "builtin"
    EXTERNAL = "external"


@dataclass(frozen=True, kw_only=True)
class CodeIdentity:
    commit: str | None
    dirty: bool | None


@dataclass(frozen=True, kw_only=True)
class FeatureCostRecord:
    """Both fields null means the dataset has unit feature costs."""

    path: str | None
    sha256: str | None


@dataclass(frozen=True, kw_only=True)
class ClassifierRecord:
    bundle_path: str
    script_name: str
    script_params: str
    dataset_key: str
    method_name: str | None
    dataset_instance_index: int
    seed: int


@dataclass(frozen=True, kw_only=True)
class ResolvedSettings:
    initializer: str
    eval_split: str
    dataset_instance_indices: list[int]
    dataset_splits: list[str]
    unmaskers: dict[str, str]
    feature_costs: dict[str, FeatureCostRecord]
    classifiers: list[ClassifierRecord]
    eval_batch_sizes: dict[str, dict[str, int]]
    forcing_policy: str


@dataclass(frozen=True, kw_only=True)
class EvaluationTableRecord:
    raw_path: str
    transformed_path: str
    raw_present: bool
    transformed_present: bool
    method_name: str
    dataset_key: str
    dataset_instance_index: int
    dataset_generation_seed: int
    eval_split: str
    initializer: str
    unmasker: str
    budget_setting: BudgetSetting
    pretrained_model_name: str | None
    pretrain_seed: int | None
    train_seed: int
    train_hard_budget: int | float | None
    train_soft_budget_param: int | float | None
    eval_seed: int
    eval_hard_budget: int | float | None
    eval_soft_budget_param: int | float | None
    forced_acquisition: bool
    classifier_bundle_path: str
    eval_batch_size: int
    classifier_variants: list[ClassifierVariant] | None


@dataclass(frozen=True, kw_only=True)
class Coverage:
    datasets: list[str]
    dataset_instance_indices: list[int]
    methods: list[str]
    eval_splits: list[str]
    budget_settings: list[BudgetSetting]
    classifier_variants: list[ClassifierVariant]
    output_categories: list[str]


@dataclass(frozen=True, kw_only=True)
class ReleaseManifest:
    manifest_version: int
    release_id: str
    scope: ReleaseScope
    execution_mode: ExecutionMode
    created_at: str
    code: CodeIdentity
    workflow_config: WorkflowConfigRecord
    settings: ResolvedSettings
    coverage: Coverage
    evaluation_tables: list[EvaluationTableRecord]

    def __post_init__(self) -> None:
        if not self.release_id:
            msg = "A release manifest needs a non-empty release_id."
            raise ValueError(msg)
        if (
            self.execution_mode is ExecutionMode.SMOKE
            and self.scope is not ReleaseScope.TEST_ONLY
        ):
            msg = (
                f"Smoke-test outputs cannot be declared a {self.scope} "
                f"release ({self.release_id!r}); use scope "
                f"{ReleaseScope.TEST_ONLY}."
            )
            raise ValueError(msg)


def build_release_manifest(
    *,
    release_id: str,
    scope: ReleaseScope,
    workflow_config: WorkflowConfigRecord,
    output_root: Path,
    checkout: Path,
) -> ReleaseManifest:
    """Describe `output_root` as produced by `workflow_config`."""
    resolved = load_workflow_settings(workflow_config.merged)
    execution_mode = (
        ExecutionMode.SMOKE
        if resolved["SMOKE_TEST"]
        else ExecutionMode.PRODUCTION
    )
    tables = _evaluation_tables(resolved, output_root)
    return ReleaseManifest(
        manifest_version=MANIFEST_VERSION,
        release_id=release_id,
        scope=scope,
        execution_mode=execution_mode,
        created_at=datetime.now(UTC).isoformat(),
        code=capture_code_identity(checkout),
        workflow_config=workflow_config,
        settings=_settings(resolved, checkout),
        coverage=_coverage(tables, output_root),
        evaluation_tables=tables,
    )


def write_release_manifest(manifest: ReleaseManifest, path: Path) -> None:
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")


def read_release_manifest(path: Path) -> ReleaseManifest:
    data = json.loads(path.read_text())
    version = data.get("manifest_version")
    if version != MANIFEST_VERSION:
        msg = (
            f"Unknown release manifest version {version!r} in {path}; "
            f"this checkout reads version {MANIFEST_VERSION}."
        )
        raise ValueError(msg)
    return dacite.from_dict(
        ReleaseManifest,
        data,
        config=dacite.Config(
            cast=[
                ReleaseScope,
                ExecutionMode,
                BudgetSetting,
                ClassifierVariant,
            ],
            strict=True,
        ),
    )


def capture_code_identity(checkout: Path) -> CodeIdentity:
    commit = _git(checkout, "rev-parse", "HEAD")
    if commit is None:
        return CodeIdentity(commit=None, dirty=None)
    # Untracked files (outputs, data) do not make the code dirty.
    status = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    return CodeIdentity(commit=commit, dirty=bool(status))


def _git(checkout: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip()


def _settings(resolved: Mapping[str, Any], checkout: Path) -> ResolvedSettings:
    datasets: list[str] = list(resolved["DATASETS"])
    return ResolvedSettings(
        initializer=resolved["INITIALIZER"],
        eval_split=resolved["EVAL_DATASET_SPLIT"],
        dataset_instance_indices=list(resolved["DATASET_INSTANCE_INDICES"]),
        dataset_splits=list(DATASET_SPLITS),
        unmaskers={
            dataset: resolved["UNMASKERS"][dataset] for dataset in datasets
        },
        feature_costs={
            dataset: _feature_cost_record(checkout, dataset)
            for dataset in datasets
        },
        classifiers=_classifiers(resolved),
        eval_batch_sizes={
            method: {dataset: batch_sizes[dataset] for dataset in datasets}
            for method, batch_sizes in resolved["EVAL_BATCH_SIZES"].items()
        },
        forcing_policy=FORCING_POLICY,
    )


def _feature_cost_record(checkout: Path, dataset: str) -> FeatureCostRecord:
    relative = FEATURE_COSTS_DIR / f"{dataset}.csv"
    path = checkout / relative
    if not path.is_file():
        return FeatureCostRecord(path=None, sha256=None)
    return FeatureCostRecord(
        path=relative.as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _classifiers(resolved: Mapping[str, Any]) -> list[ClassifierRecord]:
    tag = _initializer_tag(resolved)
    records: list[ClassifierRecord] = []
    for dataset in resolved["DATASETS"]:
        classifier_config = resolved["CLASSIFIER_NAMES"][dataset]
        if isinstance(classifier_config, dict):
            script_name = classifier_config["script_name"]
            script_params = " ".join(
                classifier_config.get("script_params", [])
            )
        else:
            script_name, script_params = classifier_config, ""
        records.append(
            ClassifierRecord(
                bundle_path=f"trained_classifiers/{tag}/dataset-{dataset}.bundle",
                script_name=script_name,
                script_params=script_params,
                dataset_key=dataset,
                method_name=None,
                dataset_instance_index=CLASSIFIER_DATASET_INSTANCE_INDEX,
                seed=CLASSIFIER_SEED,
            )
        )
    for method, script_name in resolved[
        "METHOD_CLASSIFIER_SCRIPT_NAMES"
    ].items():
        records.extend(
            ClassifierRecord(
                bundle_path=_classifier_bundle_path(resolved, method, dataset),
                script_name=script_name,
                script_params=resolved["METHOD_CLASSIFIER_SCRIPT_PARAMS"][
                    method
                ],
                dataset_key=dataset,
                method_name=method,
                dataset_instance_index=CLASSIFIER_DATASET_INSTANCE_INDEX,
                seed=CLASSIFIER_SEED,
            )
            for dataset in resolved["DATASETS"]
        )
    return records


def _classifier_bundle_path(
    resolved: Mapping[str, Any], method: str, dataset: str
) -> str:
    # Mirrors `_classifier_bundle_for_method` in rules/evaluation.smk.
    tag = _initializer_tag(resolved)
    if method in resolved["METHOD_CLASSIFIER_SCRIPT_NAMES"]:
        return f"trained_classifiers/{tag}/method-{method}+dataset-{dataset}.bundle"
    return f"trained_classifiers/{tag}/dataset-{dataset}.bundle"


def _initializer_tag(resolved: Mapping[str, Any]) -> str:
    return f"initializer-{resolved['INITIALIZER']}"


def _evaluation_tables(
    resolved: Mapping[str, Any], output_root: Path
) -> list[EvaluationTableRecord]:
    # Mirrors the targets of `all_eval_methods` in rules/helpers.smk and
    # `transform_eval_data` in rules/transformations.smk; the workflow test
    # `test_release_manifest_tables_match_workflow_targets` pins the two.
    split = resolved["EVAL_DATASET_SPLIT"]
    tag = _initializer_tag(resolved)
    records: list[EvaluationTableRecord] = []
    for method in resolved["METHODS"]:
        pretrained_model = resolved["METHOD_TO_PRETRAINED_MODEL"].get(method)
        for dataset in resolved["DATASETS"]:
            for index in resolved["DATASET_INSTANCE_INDICES"]:
                pretrain_folder = (
                    f"{resolved['NO_PRETRAIN_STR']}/"
                    if pretrained_model is None
                    else f"pretrain_seed-{index}/"
                )
                for (
                    train_hard_budget,
                    eval_hard_budget,
                    train_soft_budget_param,
                    eval_soft_budget_param,
                ) in resolved["BUDGET_PARAMS"][method][dataset]:
                    relative = (
                        f"eval_split-{split}/{tag}/{method}/"
                        f"dataset-{dataset}+instance_idx-{index}/"
                        f"{pretrain_folder}"
                        f"train_seed-{index}+"
                        f"train_hard_budget-{train_hard_budget}+"
                        f"train_soft_budget_param-{train_soft_budget_param}/"
                        f"eval_seed-{index}+"
                        f"eval_hard_budget-{eval_hard_budget}+"
                        f"eval_soft_budget_param-{eval_soft_budget_param}/"
                        "eval_data.parquet"
                    )
                    raw_path = f"eval_results/{relative}"
                    transformed_path = f"eval_results_transformed/{relative}"
                    eval_hard = _nullable(eval_hard_budget)
                    records.append(
                        EvaluationTableRecord(
                            raw_path=raw_path,
                            transformed_path=transformed_path,
                            raw_present=(output_root / raw_path).is_file(),
                            transformed_present=(
                                output_root / transformed_path
                            ).is_file(),
                            method_name=method,
                            dataset_key=dataset,
                            dataset_instance_index=index,
                            dataset_generation_seed=index,
                            eval_split=split,
                            initializer=resolved["INITIALIZER"],
                            unmasker=resolved["UNMASKERS"][dataset],
                            budget_setting=(
                                BudgetSetting.SOFT_BUDGET
                                if eval_hard is None
                                else BudgetSetting.HARD_BUDGET
                            ),
                            pretrained_model_name=pretrained_model,
                            pretrain_seed=(
                                None if pretrained_model is None else index
                            ),
                            train_seed=index,
                            train_hard_budget=_nullable(train_hard_budget),
                            train_soft_budget_param=_nullable(
                                train_soft_budget_param
                            ),
                            eval_seed=index,
                            eval_hard_budget=eval_hard,
                            eval_soft_budget_param=_nullable(
                                eval_soft_budget_param
                            ),
                            forced_acquisition=eval_hard is not None,
                            classifier_bundle_path=_classifier_bundle_path(
                                resolved, method, dataset
                            ),
                            eval_batch_size=resolved["EVAL_BATCH_SIZES"][
                                method
                            ][dataset],
                            classifier_variants=_classifier_variants(
                                output_root / raw_path
                            ),
                        )
                    )
    return records


def _nullable(value: float | str) -> int | float | None:
    if value == "null":
        return None
    if isinstance(value, str):
        msg = f"Budget value is neither a number nor 'null': {value!r}"
        raise TypeError(msg)
    return value


def _classifier_variants(raw_table: Path) -> list[ClassifierVariant] | None:
    if not raw_table.is_file():
        return None
    columns = {
        ClassifierVariant.BUILTIN: "builtin_predicted_class",
        ClassifierVariant.EXTERNAL: "external_predicted_class",
    }
    table = pq.read_table(raw_table, columns=list(columns.values()))
    return [
        variant
        for variant, column in columns.items()
        if table.column(column).null_count < table.num_rows
    ]


def _coverage(
    tables: list[EvaluationTableRecord], output_root: Path
) -> Coverage:
    present = [
        table
        for table in tables
        if table.raw_present or table.transformed_present
    ]
    return Coverage(
        datasets=sorted({table.dataset_key for table in present}),
        dataset_instance_indices=sorted(
            {table.dataset_instance_index for table in present}
        ),
        methods=sorted({table.method_name for table in present}),
        eval_splits=sorted({table.eval_split for table in present}),
        budget_settings=sorted({table.budget_setting for table in present}),
        classifier_variants=sorted(
            {
                variant
                for table in present
                for variant in table.classifier_variants or []
            }
        ),
        output_categories=sorted(
            category.name
            for category in output_root.iterdir()
            if category.is_dir()
            and any(path.is_file() for path in category.rglob("*"))
        )
        if output_root.is_dir()
        else [],
    )
