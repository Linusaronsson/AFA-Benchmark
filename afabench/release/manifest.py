"""
The release manifest: identity, redistribution review and artifact index.

A release manifest is a JSON file written beside an output snapshot's
`output/` tree. It holds the facts that belong to the release (its id,
declared scope and the maintainers' dataset redistribution review) and an
index of the snapshot's artifacts, generated from their provenance records
and never from the workflow configuration. That configuration is recorded
as the maintainer declares it, for users who lay out a comparison against
the release
(`docs/adr/0006-release-manifest-indexes-artifact-provenance.md`). The index
is a cache of the records: rebuilding it from the same output tree gives the
same result. The field list is documented in
`docs/reference/release_manifest.md`.

Bundles are found anywhere under the output root, evaluation tables under
the folders that hold the raw and transformed tables. An index entry links
its inputs to bundles by content hash, never by path.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import dacite
import pyarrow.parquet as pq
import yaml

from afabench.core.bundle_system.bundle import read_manifest
from afabench.core.provenance import ProvenanceRecord, provenance_from_manifest
from afabench.evaluation.provenance import evaluation_table_provenance
from afabench.evaluation.schemas import IDENTITY_DTYPES
from afabench.release.workflow_config import WorkflowConfigRecord

MANIFEST_VERSION = 2
RELEASE_MANIFEST_FILENAME = "release_manifest.json"
# Maintainers' redistribution reviews, relative to the checkout. A dataset
# key it does not list is unreviewed.
DATASET_REDISTRIBUTION_FILE = Path(
    "extra/conf/release/dataset_redistribution.yaml"
)


class ReleaseScope(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    SMOKE = "smoke"


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


class Stage(StrEnum):
    """The pipeline stages that write a provenance record, in order."""

    DATASET_GENERATION = "dataset_generation"
    CLASSIFIER_TRAINING = "classifier_training"
    PRETRAINING = "pretraining"
    TRAINING = "training"
    EVALUATION = "evaluation"


class InputRole(StrEnum):
    """The role names of ADR 0002's provenance `inputs`."""

    TRAIN_DATASET = "train_dataset"
    VAL_DATASET = "val_dataset"
    EVAL_DATASET = "eval_dataset"
    CLASSIFIER = "classifier"
    PRETRAINED_MODEL = "pretrained_model"
    METHOD = "method"


BUNDLE_CATEGORIES = {
    Stage.DATASET_GENERATION: PayloadCategory.DATASET_BUNDLE,
    Stage.CLASSIFIER_TRAINING: PayloadCategory.CLASSIFIER_BUNDLE,
    Stage.PRETRAINING: PayloadCategory.PRETRAINED_MODEL_BUNDLE,
    Stage.TRAINING: PayloadCategory.AFA_METHOD_BUNDLE,
}
# The payload category of the bundle an input of each role is
INPUT_CATEGORIES = {
    InputRole.TRAIN_DATASET: PayloadCategory.DATASET_BUNDLE,
    InputRole.VAL_DATASET: PayloadCategory.DATASET_BUNDLE,
    InputRole.EVAL_DATASET: PayloadCategory.DATASET_BUNDLE,
    InputRole.CLASSIFIER: PayloadCategory.CLASSIFIER_BUNDLE,
    InputRole.PRETRAINED_MODEL: PayloadCategory.PRETRAINED_MODEL_BUNDLE,
    InputRole.METHOD: PayloadCategory.AFA_METHOD_BUNDLE,
}
# Top-level folders of the output root holding each evaluation table kind
EVALUATION_TABLE_FOLDERS = {
    PayloadCategory.RAW_EVALUATION_TABLE: "eval_results",
    PayloadCategory.TRANSFORMED_EVALUATION_TABLE: "eval_results_transformed",
}
PREDICTION_COLUMNS = {
    ClassifierVariant.BUILTIN: "builtin_predicted_class",
    ClassifierVariant.EXTERNAL: "external_predicted_class",
}


@dataclass(frozen=True, kw_only=True)
class CodeIdentity:
    """The code that produced an artifact; null is unknown."""

    commit: str | None
    dirty: bool | None

    @property
    def clean(self) -> bool:
        return self.commit is not None and self.dirty is False


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
class IndexedInput:
    """
    One input of an artifact, as its record names it.

    `path` is recorded as given to the producing job, so it locates
    nothing in a release; the input is the bundle with `content_hash`.
    """

    role: InputRole
    path: str
    content_hash: str | None


@dataclass(frozen=True, kw_only=True)
class BundleEntry:
    """One bundle in the output tree, described by its record."""

    path: str
    category: PayloadCategory
    class_name: str
    content_hash: str | None
    size_bytes: int
    stage: Stage
    code: CodeIdentity
    smoke_test: bool
    seed: int
    method_name: str | None
    dataset_key: str | None
    dataset_realization_index: int | None
    split: str | None
    inputs: list[IndexedInput]


@dataclass(frozen=True, kw_only=True)
class EvaluationEntry:
    """
    One evaluation: its raw and transformed tables, either possibly absent.

    Identity comes from the tables' identity columns; a null value is
    unknown, as in the columns. `classifier_variants` are the variants with
    a non-null prediction.
    """

    raw_path: str | None
    transformed_path: str | None
    raw_size_bytes: int | None
    transformed_size_bytes: int | None
    code: CodeIdentity
    smoke_test: bool
    method_name: str | None
    dataset_key: str | None
    dataset_realization_index: int | None
    eval_split: str | None
    initializer: str | None
    budget_setting: BudgetSetting
    train_seed: int | None
    train_hard_budget: float | None
    train_soft_budget_param: float | None
    eval_seed: int | None
    eval_hard_budget: float | None
    eval_soft_budget_param: float | None
    classifier_variants: list[ClassifierVariant]
    inputs: list[IndexedInput]

    @property
    def path(self) -> str:
        """The path naming this evaluation: its raw table, if present."""
        path = self.raw_path or self.transformed_path
        if path is None:
            msg = "An evaluation entry has neither a raw nor a transformed table."
            raise ValueError(msg)
        return path


@dataclass(frozen=True, kw_only=True)
class PayloadCoverage:
    """
    How many payloads of one category the release holds.

    `class_names` are the bundle classes present, so the loaders a restored
    category needs; empty for evaluation tables.
    """

    category: PayloadCategory
    count: int
    size_bytes: int
    class_names: list[str]


@dataclass(frozen=True, kw_only=True)
class Coverage:
    datasets: list[str]
    dataset_realization_indices: list[int]
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
    workflow_config: WorkflowConfigRecord
    dataset_redistribution: dict[str, DatasetRedistribution]
    coverage: Coverage
    evaluations: list[EvaluationEntry]
    bundles: list[BundleEntry]

    def __post_init__(self) -> None:
        if not self.release_id:
            msg = "A release manifest needs a non-empty release_id."
            raise ValueError(msg)
        if (
            self.execution_mode is ExecutionMode.SMOKE
            and self.scope is not ReleaseScope.SMOKE
        ):
            msg = (
                f"Smoke-test outputs cannot be declared a {self.scope} "
                f"release ({self.release_id!r}); use scope "
                f"{ReleaseScope.SMOKE}."
            )
            raise ValueError(msg)


@dataclass(frozen=True, kw_only=True)
class ArtifactIndex:
    """
    The artifacts of an output tree, and those it cannot describe.

    `unrecorded` are the bundles and evaluation tables without a provenance
    record, as paths relative to the output root.
    """

    evaluations: list[EvaluationEntry]
    bundles: list[BundleEntry]
    unrecorded: list[str]


@dataclass(frozen=True, kw_only=True)
class DanglingInput:
    """An input whose bundle, by content hash, is not in the release."""

    artifact_path: str
    role: InputRole
    path: str


@dataclass(frozen=True, kw_only=True)
class StageCode:
    """How many artifacts of one stage one code identity produced."""

    stage: Stage
    code: CodeIdentity
    artifacts: int


class UnrecordedArtifactsError(ValueError):
    """Raised when a release would ship artifacts without a record."""


def build_release_manifest(
    *,
    release_id: str,
    scope: ReleaseScope,
    workflow_config: WorkflowConfigRecord,
    output_root: Path,
    checkout: Path,
) -> ReleaseManifest:
    """
    Describe `output_root` by its artifacts' provenance records.

    `workflow_config` is the maintainer's declaration of the configuration
    whose targets lay out the release; it is recorded, never read.
    `checkout` supplies the maintainers' dataset redistribution review.
    """
    index = index_artifacts(output_root)
    if index.unrecorded:
        msg = (
            f"Release {release_id!r} cannot describe these artifacts, which "
            "have no provenance record; regenerate or remove them:\n  "
            + "\n  ".join(index.unrecorded)
        )
        raise UnrecordedArtifactsError(msg)
    if not index.evaluations and not index.bundles:
        msg = f"Release {release_id!r} holds no artifact under {output_root}."
        raise ValueError(msg)
    datasets = sorted(
        {
            entry.dataset_key
            for entry in [*index.evaluations, *index.bundles]
            if entry.dataset_key is not None
        }
    )
    return ReleaseManifest(
        manifest_version=MANIFEST_VERSION,
        release_id=release_id,
        scope=scope,
        execution_mode=_execution_mode(index),
        created_at=datetime.now(UTC).isoformat(),
        workflow_config=workflow_config,
        dataset_redistribution=_dataset_redistribution(checkout, datasets),
        coverage=_coverage(index, output_root),
        evaluations=index.evaluations,
        bundles=index.bundles,
    )


def index_artifacts(output_root: Path) -> ArtifactIndex:
    """Index every bundle and evaluation table under `output_root`."""
    if not output_root.is_dir():
        msg = f"Output root does not exist: {output_root}"
        raise FileNotFoundError(msg)
    unrecorded: list[str] = []
    bundles: list[BundleEntry] = []
    for path in sorted(output_root.rglob("*.bundle")):
        if not path.is_dir():
            continue
        relative = path.relative_to(output_root).as_posix()
        entry = _bundle_entry(path, relative)
        if entry is None:
            unrecorded.append(relative)
        else:
            bundles.append(entry)
    tables: dict[PayloadCategory, list[tuple[str, ProvenanceRecord]]] = {}
    for category, folder in EVALUATION_TABLE_FOLDERS.items():
        tables[category] = []
        for path in sorted((output_root / folder).rglob("*.parquet")):
            relative = path.relative_to(output_root).as_posix()
            record = evaluation_table_provenance(path)
            if record is None:
                unrecorded.append(relative)
            else:
                tables[category].append((relative, record))
    return ArtifactIndex(
        evaluations=_evaluation_entries(output_root, tables),
        bundles=bundles,
        unrecorded=unrecorded,
    )


def dangling_inputs(manifest: ReleaseManifest) -> list[DanglingInput]:
    """Return the inputs no bundle of the release matches by hash."""
    hashes = {bundle.content_hash for bundle in manifest.bundles}
    artifacts: list[tuple[str, list[IndexedInput]]] = [
        (bundle.path, bundle.inputs) for bundle in manifest.bundles
    ]
    artifacts.extend(
        (evaluation.path, evaluation.inputs)
        for evaluation in manifest.evaluations
    )
    return [
        DanglingInput(artifact_path=path, role=entry.role, path=entry.path)
        for path, inputs in artifacts
        for entry in inputs
        if entry.content_hash is None or entry.content_hash not in hashes
    ]


def code_by_stage(manifest: ReleaseManifest) -> list[StageCode]:
    """Count the release's artifacts per stage and producing code."""
    counts: Counter[tuple[Stage, CodeIdentity]] = Counter(
        (bundle.stage, bundle.code) for bundle in manifest.bundles
    )
    counts.update(
        (Stage.EVALUATION, evaluation.code)
        for evaluation in manifest.evaluations
    )
    stages = list(Stage)
    return [
        StageCode(stage=stage, code=code, artifacts=artifacts)
        for (stage, code), artifacts in sorted(
            counts.items(),
            key=lambda item: (
                stages.index(item[0][0]),
                item[0][1].commit or "",
                str(item[0][1].dirty),
            ),
        )
    ]


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
                Stage,
                InputRole,
                RedistributionStatus,
            ],
            strict=True,
        ),
    )


def payload_coverage(index: ArtifactIndex) -> list[PayloadCoverage]:
    """Count and size each payload category of an index."""
    payloads = [
        PayloadCoverage(
            category=PayloadCategory.RAW_EVALUATION_TABLE,
            count=sum(e.raw_path is not None for e in index.evaluations),
            size_bytes=sum(e.raw_size_bytes or 0 for e in index.evaluations),
            class_names=[],
        ),
        PayloadCoverage(
            category=PayloadCategory.TRANSFORMED_EVALUATION_TABLE,
            count=sum(
                e.transformed_path is not None for e in index.evaluations
            ),
            size_bytes=sum(
                e.transformed_size_bytes or 0 for e in index.evaluations
            ),
            class_names=[],
        ),
    ]
    for category in BUNDLE_CATEGORIES.values():
        bundles = [b for b in index.bundles if b.category is category]
        payloads.append(
            PayloadCoverage(
                category=category,
                count=len(bundles),
                size_bytes=sum(bundle.size_bytes for bundle in bundles),
                class_names=sorted({bundle.class_name for bundle in bundles}),
            )
        )
    return payloads


def execution_mode(index: ArtifactIndex) -> ExecutionMode | None:
    """Return the execution mode the records agree on; null if none."""
    return (
        _execution_mode(index) if index.bundles or index.evaluations else None
    )


def _execution_mode(index: ArtifactIndex) -> ExecutionMode:
    # Dataset generation has no smoke mode and always records production,
    # so one smoke artifact makes the tree a smoke tree.
    if any(entry.smoke_test for entry in [*index.bundles, *index.evaluations]):
        return ExecutionMode.SMOKE
    return ExecutionMode.PRODUCTION


def _bundle_entry(path: Path, relative: str) -> BundleEntry | None:
    if not (path / "manifest.json").is_file():
        return None
    bundle_manifest = read_manifest(path)
    record = provenance_from_manifest(bundle_manifest)
    if record is None:
        return None
    stage = Stage(record.stage)
    if stage not in BUNDLE_CATEGORIES:
        msg = (
            f"Bundle {relative} records stage {stage}, which writes no bundle."
        )
        raise ValueError(msg)
    return BundleEntry(
        path=relative,
        category=BUNDLE_CATEGORIES[stage],
        class_name=bundle_manifest["class_name"],
        content_hash=bundle_manifest.get("content_hash"),
        size_bytes=sum(
            file.stat().st_size for file in path.rglob("*") if file.is_file()
        ),
        stage=stage,
        code=_code(record),
        smoke_test=record.smoke_test,
        seed=record.seed,
        method_name=record.method_name,
        dataset_key=record.dataset_key,
        dataset_realization_index=record.dataset_realization_index,
        split=record.split,
        inputs=_inputs(record),
    )


def _evaluation_entries(
    output_root: Path,
    tables: dict[PayloadCategory, list[tuple[str, ProvenanceRecord]]],
) -> list[EvaluationEntry]:
    # The transform copies the raw table's record, so equal records are one
    # evaluation.
    paired: dict[str, dict[PayloadCategory, str]] = {}
    records: dict[str, ProvenanceRecord] = {}
    for category, entries in tables.items():
        for relative, record in entries:
            key = json.dumps(record.to_json_dict(), sort_keys=True)
            if category in paired.setdefault(key, {}):
                msg = (
                    f"{paired[key][category]} and {relative} hold the same "
                    "provenance record; one evaluation has one table of "
                    "each kind."
                )
                raise ValueError(msg)
            paired[key][category] = relative
            records[key] = record
    return [
        _evaluation_entry(
            output_root,
            records[key],
            raw_path=paths.get(PayloadCategory.RAW_EVALUATION_TABLE),
            transformed_path=paths.get(
                PayloadCategory.TRANSFORMED_EVALUATION_TABLE
            ),
        )
        for key, paths in sorted(
            paired.items(),
            key=lambda item: _evaluation_sort_key(item[1]),
        )
    ]


def _evaluation_sort_key(paths: dict[PayloadCategory, str]) -> str:
    return paths.get(PayloadCategory.RAW_EVALUATION_TABLE) or paths.get(
        PayloadCategory.TRANSFORMED_EVALUATION_TABLE, ""
    )


def _evaluation_entry(
    output_root: Path,
    record: ProvenanceRecord,
    *,
    raw_path: str | None,
    transformed_path: str | None,
) -> EvaluationEntry:
    # Identity is constant per table, so either table's first row holds it.
    table_path = output_root / (raw_path or transformed_path or "")
    identity = _identity(table_path)
    variants = (
        _raw_classifier_variants(output_root / raw_path)
        if raw_path is not None
        else _transformed_classifier_variants(table_path)
    )
    eval_hard_budget = identity.get("eval_hard_budget")
    return EvaluationEntry(
        raw_path=raw_path,
        transformed_path=transformed_path,
        raw_size_bytes=_file_size(output_root, raw_path),
        transformed_size_bytes=_file_size(output_root, transformed_path),
        code=_code(record),
        smoke_test=record.smoke_test,
        method_name=identity.get("afa_method"),
        dataset_key=identity.get("dataset"),
        dataset_realization_index=identity.get("dataset_realization_index"),
        eval_split=identity.get("eval_split"),
        initializer=identity.get("initializer"),
        budget_setting=(
            BudgetSetting.SOFT_BUDGET
            if eval_hard_budget is None
            else BudgetSetting.HARD_BUDGET
        ),
        train_seed=identity.get("train_seed"),
        train_hard_budget=identity.get("train_hard_budget"),
        train_soft_budget_param=identity.get("train_soft_budget_param"),
        eval_seed=identity.get("eval_seed"),
        eval_hard_budget=eval_hard_budget,
        eval_soft_budget_param=identity.get("eval_soft_budget_param"),
        classifier_variants=variants,
        inputs=_inputs(record),
    )


def _identity(path: Path) -> dict[str, Any]:
    """Read the first row's identity columns; a missing one is null."""
    parquet = pq.ParquetFile(path)
    columns = [
        column
        for column in IDENTITY_DTYPES
        if column in parquet.schema_arrow.names
    ]
    table = parquet.read(columns=columns).slice(0, 1)
    if table.num_rows == 0:
        return {}
    return {column: table.column(column)[0].as_py() for column in columns}


def _raw_classifier_variants(path: Path) -> list[ClassifierVariant]:
    table = pq.read_table(path, columns=list(PREDICTION_COLUMNS.values()))
    return [
        variant
        for variant, column in PREDICTION_COLUMNS.items()
        if table.column(column).null_count < table.num_rows
    ]


def _transformed_classifier_variants(path: Path) -> list[ClassifierVariant]:
    table = pq.read_table(path, columns=["classifier", "predicted_class"])
    predicted = table.filter(table.column("predicted_class").is_valid())
    present = set(predicted.column("classifier").unique().to_pylist())
    return [variant for variant in ClassifierVariant if variant in present]


def _file_size(output_root: Path, relative: str | None) -> int | None:
    if relative is None:
        return None
    return (output_root / relative).stat().st_size


def _code(record: ProvenanceRecord) -> CodeIdentity:
    return CodeIdentity(commit=record.code_commit, dirty=record.code_dirty)


def _inputs(record: ProvenanceRecord) -> list[IndexedInput]:
    return [
        IndexedInput(
            role=InputRole(entry.role),
            path=entry.path,
            content_hash=entry.content_hash,
        )
        for entry in record.inputs
    ]


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


def _coverage(index: ArtifactIndex, output_root: Path) -> Coverage:
    evaluations = index.evaluations

    def values[T](name: str) -> list[T]:
        return sorted(
            {
                getattr(evaluation, name)
                for evaluation in evaluations
                if getattr(evaluation, name) is not None
            }
        )

    return Coverage(
        datasets=values("dataset_key"),
        dataset_realization_indices=values("dataset_realization_index"),
        methods=values("method_name"),
        eval_splits=values("eval_split"),
        budget_settings=values("budget_setting"),
        classifier_variants=sorted(
            {
                variant
                for evaluation in evaluations
                for variant in evaluation.classifier_variants
            }
        ),
        output_categories=sorted(
            category.name
            for category in output_root.iterdir()
            if category.is_dir()
            and any(path.is_file() for path in category.rglob("*"))
        ),
        payloads=payload_coverage(index),
    )
