"""
The release manifest: identity, provenance and coverage of a snapshot.

A release manifest is a JSON file written beside an output snapshot's
`output/` tree. It is filled from the workflow configuration and the git
state of the checkout. Per-table and per-bundle identity is enumerated
forward from the resolved workflow configuration, the same way the
workflow's `all_*` rules name their targets, rather than parsed back out of
paths. Once evaluation tables and bundles carry their own identity columns
and provenance record (`docs/adr/0002-provenance-recorded-in-artifacts.md`),
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
import yaml

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
# Maintainers' redistribution reviews, relative to the checkout. A dataset
# key it does not list is unreviewed.
DATASET_REDISTRIBUTION_FILE = Path(
    "extra/conf/release/dataset_redistribution.yaml"
)


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


class RedistributionStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    PERMITTED = "permitted"
    RESTRICTED = "restricted"


class PayloadCategory(StrEnum):
    RAW_EVALUATION_TABLE = "raw_evaluation_table"
    TRANSFORMED_EVALUATION_TABLE = "transformed_evaluation_table"
    DATASET_BUNDLE = "dataset_bundle"
    CLASSIFIER_BUNDLE = "classifier_bundle"
    PRETRAINED_MODEL_BUNDLE = "pretrained_model_bundle"
    AFA_METHOD_BUNDLE = "afa_method_bundle"


class InputRole(StrEnum):
    """The role names of ADR 0002's provenance `inputs`."""

    TRAIN_DATASET = "train_dataset"
    VAL_DATASET = "val_dataset"
    EVAL_DATASET = "eval_dataset"
    CLASSIFIER = "classifier"
    PRETRAINED_MODEL = "pretrained_model"
    METHOD = "method"


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
class DatasetRedistribution:
    """A maintainer's review of whether a dataset's bundles may be public."""

    status: RedistributionStatus
    license: str | None
    source: str | None
    reviewed_by: str | None
    notes: str | None


UNREVIEWED_REDISTRIBUTION = DatasetRedistribution(
    status=RedistributionStatus.UNREVIEWED,
    license=None,
    source=None,
    reviewed_by=None,
    notes=None,
)


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
    dataset_redistribution: dict[str, DatasetRedistribution]
    classifiers: list[ClassifierRecord]
    eval_batch_sizes: dict[str, dict[str, int]]
    forcing_policy: str


@dataclass(frozen=True, kw_only=True)
class BundleInput:
    role: InputRole
    path: str


@dataclass(frozen=True, kw_only=True)
class EvaluationTableRecord:
    raw_path: str
    transformed_path: str
    raw_present: bool
    transformed_present: bool
    raw_size_bytes: int | None
    transformed_size_bytes: int | None
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
    inputs: list[BundleInput]


@dataclass(frozen=True, kw_only=True)
class BundleRecord:
    """
    One bundle the workflow config schedules, present or not.

    `method_name` is null for shared prerequisites: dataset splits, the
    external classifier and pretrained models. `bundle_manifest` is the
    bundle's own `manifest.json`, null if the bundle is absent or has none.
    """

    path: str
    category: PayloadCategory
    present: bool
    size_bytes: int | None
    dataset_key: str
    dataset_instance_index: int
    split: str | None
    method_name: str | None
    pretrained_model_name: str | None
    seed: int
    train_hard_budget: int | float | None
    train_soft_budget_param: int | float | None
    inputs: list[BundleInput]
    bundle_manifest: dict[str, Any] | None


@dataclass(frozen=True, kw_only=True)
class PayloadCoverage:
    """
    How many payloads of one category the config schedules and has.

    `class_names` are the bundle classes present, so the loaders a restored
    category needs; empty for evaluation tables.
    """

    category: PayloadCategory
    scheduled: int
    present: int
    size_bytes: int
    class_names: list[str]


@dataclass(frozen=True, kw_only=True)
class Coverage:
    datasets: list[str]
    dataset_instance_indices: list[int]
    methods: list[str]
    eval_splits: list[str]
    budget_settings: list[BudgetSetting]
    classifier_variants: list[ClassifierVariant]
    output_categories: list[str]
    payloads: list[PayloadCoverage]


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
    bundles: list[BundleRecord]

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
    execution_mode = _execution_mode(resolved)
    tables = _evaluation_tables(resolved, output_root)
    bundles = _bundles(resolved, output_root)
    return ReleaseManifest(
        manifest_version=MANIFEST_VERSION,
        release_id=release_id,
        scope=scope,
        execution_mode=execution_mode,
        created_at=datetime.now(UTC).isoformat(),
        code=capture_code_identity(checkout),
        workflow_config=workflow_config,
        settings=_settings(resolved, checkout),
        coverage=_coverage(tables, bundles, output_root),
        evaluation_tables=tables,
        bundles=bundles,
    )


@dataclass(frozen=True, kw_only=True)
class PayloadInventory:
    execution_mode: ExecutionMode
    payloads: list[PayloadCoverage]


def inventory_payloads(
    *, workflow_config: WorkflowConfigRecord, output_root: Path
) -> PayloadInventory:
    """Count and size the payloads of `output_root`, as a manifest would."""
    resolved = load_workflow_settings(workflow_config.merged)
    return PayloadInventory(
        execution_mode=_execution_mode(resolved),
        payloads=_payload_coverage(
            _evaluation_tables(resolved, output_root),
            _bundles(resolved, output_root),
        ),
    )


def _execution_mode(resolved: Mapping[str, Any]) -> ExecutionMode:
    if resolved["SMOKE_TEST"]:
        return ExecutionMode.SMOKE
    return ExecutionMode.PRODUCTION


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
                PayloadCategory,
                InputRole,
                RedistributionStatus,
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
        dataset_redistribution=_dataset_redistribution(checkout, datasets),
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


def _dataset_redistribution(
    checkout: Path, datasets: list[str]
) -> dict[str, DatasetRedistribution]:
    path = checkout / DATASET_REDISTRIBUTION_FILE
    content = yaml.safe_load(path.read_text()) if path.is_file() else None
    reviews: dict[str, Any] = (content or {}).get("datasets") or {}
    records: dict[str, DatasetRedistribution] = {}
    for dataset in datasets:
        if dataset not in reviews:
            records[dataset] = UNREVIEWED_REDISTRIBUTION
            continue
        review = dacite.from_dict(
            DatasetRedistribution,
            reviews[dataset],
            config=dacite.Config(cast=[RedistributionStatus], strict=True),
        )
        if review.status is RedistributionStatus.UNREVIEWED:
            msg = (
                f"{path} lists dataset {dataset!r} as {review.status}; "
                "a review is permitted or restricted, an unreviewed "
                "dataset is left out."
            )
            raise ValueError(msg)
        records[dataset] = review
    return records


def _classifiers(resolved: Mapping[str, Any]) -> list[ClassifierRecord]:
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
                bundle_path=_classifier_bundle_path(resolved, None, dataset),
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
    resolved: Mapping[str, Any], method: str | None, dataset: str
) -> str:
    # Method `None` names the external classifier of `dataset`. Mirrors `_classifier_bundle_for_method` in rules/evaluation.smk.
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
            classifier_bundle_path = _classifier_bundle_path(
                resolved, method, dataset
            )
            for index in resolved["DATASET_INSTANCE_INDICES"]:
                for (
                    train_hard_budget,
                    eval_hard_budget,
                    train_soft_budget_param,
                    eval_soft_budget_param,
                ) in resolved["BUDGET_PARAMS"][method][dataset]:
                    relative = (
                        f"eval_split-{split}/{tag}/{method}/"
                        f"dataset-{dataset}+instance_idx-{index}/"
                        f"{_pretrain_folder(resolved, method, index)}"
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
                    raw_file = output_root / raw_path
                    transformed_file = output_root / transformed_path
                    eval_hard = _nullable(eval_hard_budget)
                    records.append(
                        EvaluationTableRecord(
                            raw_path=raw_path,
                            transformed_path=transformed_path,
                            raw_present=raw_file.is_file(),
                            transformed_present=transformed_file.is_file(),
                            raw_size_bytes=_file_size(raw_file),
                            transformed_size_bytes=_file_size(
                                transformed_file
                            ),
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
                            classifier_bundle_path=classifier_bundle_path,
                            eval_batch_size=resolved["EVAL_BATCH_SIZES"][
                                method
                            ][dataset],
                            classifier_variants=_classifier_variants(raw_file),
                            inputs=[
                                BundleInput(
                                    role=InputRole.EVAL_DATASET,
                                    path=_dataset_bundle_path(
                                        dataset, index, split
                                    ),
                                ),
                                BundleInput(
                                    role=InputRole.METHOD,
                                    path=_method_bundle_path(
                                        resolved,
                                        method,
                                        dataset,
                                        index,
                                        train_hard_budget,
                                        train_soft_budget_param,
                                    ),
                                ),
                                BundleInput(
                                    role=InputRole.CLASSIFIER,
                                    path=classifier_bundle_path,
                                ),
                            ],
                        )
                    )
    return records


def _file_size(path: Path) -> int | None:
    return path.stat().st_size if path.is_file() else None


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
    tables: list[EvaluationTableRecord],
    bundles: list[BundleRecord],
    output_root: Path,
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
        payloads=_payload_coverage(tables, bundles),
    )


def _payload_coverage(
    tables: list[EvaluationTableRecord], bundles: list[BundleRecord]
) -> list[PayloadCoverage]:
    payloads = [
        PayloadCoverage(
            category=PayloadCategory.RAW_EVALUATION_TABLE,
            scheduled=len(tables),
            present=sum(table.raw_present for table in tables),
            size_bytes=sum(table.raw_size_bytes or 0 for table in tables),
            class_names=[],
        ),
        PayloadCoverage(
            category=PayloadCategory.TRANSFORMED_EVALUATION_TABLE,
            scheduled=len(tables),
            present=sum(table.transformed_present for table in tables),
            size_bytes=sum(
                table.transformed_size_bytes or 0 for table in tables
            ),
            class_names=[],
        ),
    ]
    for category in [
        PayloadCategory.DATASET_BUNDLE,
        PayloadCategory.CLASSIFIER_BUNDLE,
        PayloadCategory.PRETRAINED_MODEL_BUNDLE,
        PayloadCategory.AFA_METHOD_BUNDLE,
    ]:
        records = [bundle for bundle in bundles if bundle.category is category]
        payloads.append(
            PayloadCoverage(
                category=category,
                scheduled=len(records),
                present=sum(bundle.present for bundle in records),
                size_bytes=sum(bundle.size_bytes or 0 for bundle in records),
                class_names=sorted(
                    {
                        bundle.bundle_manifest["class_name"]
                        for bundle in records
                        if bundle.bundle_manifest is not None
                        and "class_name" in bundle.bundle_manifest
                    }
                ),
            )
        )
    return payloads


def _bundles(
    resolved: Mapping[str, Any], output_root: Path
) -> list[BundleRecord]:
    # Mirrors the targets of `all_generate_datasets`, `all_train_classifiers`,
    # `all_pretrain_models` and `all_train_methods` in rules/helpers.smk, and
    # the inputs of the rules producing them; the workflow test
    # `test_release_manifest_bundles_match_workflow_targets` pins the two.
    datasets: list[str] = list(resolved["DATASETS"])
    indices: list[int] = list(resolved["DATASET_INSTANCE_INDICES"])
    records = [
        _bundle_record(
            output_root,
            path=_dataset_bundle_path(dataset, index, split),
            category=PayloadCategory.DATASET_BUNDLE,
            inputs=[],
            dataset_key=dataset,
            dataset_instance_index=index,
            split=split,
            seed=index,
        )
        for dataset in datasets
        for index in indices
        for split in DATASET_SPLITS
    ]
    classifier_owners = [
        None,
        *(
            method
            for method in resolved["METHODS"]
            if method in resolved["METHOD_CLASSIFIER_SCRIPT_NAMES"]
        ),
    ]
    records.extend(
        _bundle_record(
            output_root,
            path=_classifier_bundle_path(resolved, method, dataset),
            category=PayloadCategory.CLASSIFIER_BUNDLE,
            inputs=_training_inputs(
                dataset, CLASSIFIER_DATASET_INSTANCE_INDEX
            ),
            dataset_key=dataset,
            dataset_instance_index=CLASSIFIER_DATASET_INSTANCE_INDEX,
            method_name=method,
            seed=CLASSIFIER_SEED,
        )
        for method in classifier_owners
        for dataset in datasets
    )
    records.extend(
        _bundle_record(
            output_root,
            path=_pretrained_model_bundle_path(resolved, name, dataset, index),
            category=PayloadCategory.PRETRAINED_MODEL_BUNDLE,
            # Pretraining always reads the external classifier.
            inputs=[
                *_training_inputs(dataset, index),
                BundleInput(
                    role=InputRole.CLASSIFIER,
                    path=_classifier_bundle_path(resolved, None, dataset),
                ),
            ],
            dataset_key=dataset,
            dataset_instance_index=index,
            pretrained_model_name=name,
            seed=index,
        )
        for name in resolved["PRETRAIN_NAMES"]
        for dataset in resolved["DATASETS_USED_PER_PRETRAIN_NAME"][name]
        for index in indices
    )
    for method in resolved["METHODS"]:
        pretrained_model = resolved["METHOD_TO_PRETRAINED_MODEL"].get(method)
        for dataset in datasets:
            # Several evaluations can share one trained method bundle.
            train_budgets = dict.fromkeys(
                (train_hard_budget, train_soft_budget_param)
                for (
                    train_hard_budget,
                    _eval_hard_budget,
                    train_soft_budget_param,
                    _eval_soft_budget_param,
                ) in resolved["BUDGET_PARAMS"][method][dataset]
            )
            for index in indices:
                inputs = [
                    *_training_inputs(dataset, index),
                    BundleInput(
                        role=InputRole.CLASSIFIER,
                        path=_classifier_bundle_path(
                            resolved, method, dataset
                        ),
                    ),
                ]
                if pretrained_model is not None:
                    inputs.append(
                        BundleInput(
                            role=InputRole.PRETRAINED_MODEL,
                            path=_pretrained_model_bundle_path(
                                resolved, pretrained_model, dataset, index
                            ),
                        )
                    )
                records.extend(
                    _bundle_record(
                        output_root,
                        path=_method_bundle_path(
                            resolved,
                            method,
                            dataset,
                            index,
                            train_hard_budget,
                            train_soft_budget_param,
                        ),
                        category=PayloadCategory.AFA_METHOD_BUNDLE,
                        inputs=inputs,
                        dataset_key=dataset,
                        dataset_instance_index=index,
                        method_name=method,
                        pretrained_model_name=pretrained_model,
                        seed=index,
                        train_hard_budget=_nullable(train_hard_budget),
                        train_soft_budget_param=_nullable(
                            train_soft_budget_param
                        ),
                    )
                    for train_hard_budget, train_soft_budget_param in (
                        train_budgets
                    )
                )
    return records


def _bundle_record(
    output_root: Path,
    *,
    path: str,
    category: PayloadCategory,
    inputs: list[BundleInput],
    dataset_key: str,
    dataset_instance_index: int,
    seed: int,
    split: str | None = None,
    method_name: str | None = None,
    pretrained_model_name: str | None = None,
    train_hard_budget: float | None = None,
    train_soft_budget_param: float | None = None,
) -> BundleRecord:
    bundle = output_root / path
    manifest_path = bundle / "manifest.json"
    return BundleRecord(
        path=path,
        category=category,
        present=bundle.is_dir(),
        size_bytes=(
            sum(
                file.stat().st_size
                for file in bundle.rglob("*")
                if file.is_file()
            )
            if bundle.is_dir()
            else None
        ),
        dataset_key=dataset_key,
        dataset_instance_index=dataset_instance_index,
        split=split,
        method_name=method_name,
        pretrained_model_name=pretrained_model_name,
        seed=seed,
        train_hard_budget=train_hard_budget,
        train_soft_budget_param=train_soft_budget_param,
        inputs=inputs,
        bundle_manifest=(
            json.loads(manifest_path.read_text())
            if manifest_path.is_file()
            else None
        ),
    )


def _training_inputs(dataset: str, index: int) -> list[BundleInput]:
    return [
        BundleInput(
            role=InputRole.TRAIN_DATASET,
            path=_dataset_bundle_path(dataset, index, "train"),
        ),
        BundleInput(
            role=InputRole.VAL_DATASET,
            path=_dataset_bundle_path(dataset, index, "val"),
        ),
    ]


def _dataset_bundle_path(dataset: str, index: int, split: str) -> str:
    return f"datasets/{dataset}/{index}/{split}.bundle"


def _pretrained_model_bundle_path(
    resolved: Mapping[str, Any], name: str, dataset: str, index: int
) -> str:
    return (
        f"pretrained_models/{_initializer_tag(resolved)}/{name}/"
        f"dataset-{dataset}+instance_idx-{index}/"
        f"pretrain_seed-{index}/model.bundle"
    )


def _method_bundle_path(
    resolved: Mapping[str, Any],
    method: str,
    dataset: str,
    index: int,
    train_hard_budget: float | str,
    train_soft_budget_param: float | str,
) -> str:
    return (
        f"trained_methods/{_initializer_tag(resolved)}/{method}/"
        f"dataset-{dataset}+instance_idx-{index}/"
        f"{_pretrain_folder(resolved, method, index)}"
        f"train_seed-{index}+"
        f"train_hard_budget-{train_hard_budget}+"
        f"train_soft_budget_param-{train_soft_budget_param}/"
        "method.bundle"
    )


def _pretrain_folder(
    resolved: Mapping[str, Any], method: str, index: int
) -> str:
    if method in resolved["METHOD_TO_PRETRAINED_MODEL"]:
        return f"pretrain_seed-{index}/"
    return f"{resolved['NO_PRETRAIN_STR']}/"
