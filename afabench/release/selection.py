"""
Choose which payloads of a benchmark release a download fetches.

Selection reads only the release manifest. Evaluation tables are matched
against the requested coverage; the bundles they depend on are found by
following the `inputs` of those tables and, in turn, of the bundles, then
kept if their payload category is requested. So asking for evaluation
tables alone fetches no bundle, and asking for dataset and classifier
bundles fetches the shared prerequisites of the selected evaluations
without their AFA-method bundles. Output categories (top-level folders of
the output root such as `plot_results`) are not described per file by the
manifest, so they are selected whole.

Requested coverage the release does not have is reported in `missing`,
never filled from elsewhere.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from afabench.release.manifest import (
    BudgetSetting,
    BundleRecord,
    ClassifierVariant,
    EvaluationTableRecord,
    PayloadCategory,
    ReleaseManifest,
)


@dataclass(frozen=True, kw_only=True)
class ReleaseSelection:
    """
    What to download from one release.

    Empty coverage lists do not restrict; coverage never restricts output
    categories. A table matches a classifier variant if its raw table holds
    that variant's predictions, so a table without its raw table matches
    none.
    """

    payload_categories: list[PayloadCategory] = field(default_factory=list)
    output_categories: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    dataset_instance_indices: list[int] = field(default_factory=list)
    eval_splits: list[str] = field(default_factory=list)
    initializers: list[str] = field(default_factory=list)
    budget_settings: list[BudgetSetting] = field(default_factory=list)
    classifier_variants: list[ClassifierVariant] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class SelectedPayloads:
    """Paths relative to the release's output root."""

    files: list[str]
    folders: list[str]
    missing: list[str]


type TableValues = Callable[[EvaluationTableRecord], Sequence[object]]


def select_payloads(
    manifest: ReleaseManifest, selection: ReleaseSelection
) -> SelectedPayloads:
    dimensions: list[tuple[str, Sequence[object], TableValues]] = [
        ("dataset", selection.datasets, lambda t: [t.dataset_key]),
        ("method", selection.methods, lambda t: [t.method_name]),
        (
            "dataset instance",
            selection.dataset_instance_indices,
            lambda t: [t.dataset_instance_index],
        ),
        ("eval split", selection.eval_splits, lambda t: [t.eval_split]),
        ("initializer", selection.initializers, lambda t: [t.initializer]),
        (
            "budget setting",
            selection.budget_settings,
            lambda t: [t.budget_setting],
        ),
        (
            "classifier variant",
            selection.classifier_variants,
            lambda t: t.classifier_variants or [],
        ),
    ]
    tables = [
        table
        for table in manifest.evaluation_tables
        if all(
            not wanted or set(values(table)) & set(wanted)
            for _, wanted, values in dimensions
        )
    ]
    missing = [
        f"{name} '{value}': no evaluation of the release matches"
        for name, wanted, values in dimensions
        for value in wanted
        if not any(value in values(table) for table in tables)
    ]
    files: list[str] = []
    folders: list[str] = []
    categories = set(selection.payload_categories)

    def take(
        category: str, path: str, *, present: bool, into: list[str]
    ) -> None:
        if present:
            into.append(path)
        else:
            missing.append(f"{category} {path}: not in the release")

    for table in tables:
        if PayloadCategory.RAW_EVALUATION_TABLE in categories:
            take(
                PayloadCategory.RAW_EVALUATION_TABLE,
                table.raw_path,
                present=table.raw_present,
                into=files,
            )
        if PayloadCategory.TRANSFORMED_EVALUATION_TABLE in categories:
            take(
                PayloadCategory.TRANSFORMED_EVALUATION_TABLE,
                table.transformed_path,
                present=table.transformed_present,
                into=files,
            )
    for bundle in _dependencies(manifest, tables):
        if bundle.category in categories:
            take(
                bundle.category,
                bundle.path,
                present=bundle.present,
                into=folders,
            )
    for output_category in dict.fromkeys(selection.output_categories):
        take(
            "output category",
            output_category,
            present=output_category in manifest.coverage.output_categories,
            into=folders,
        )
    return SelectedPayloads(
        files=list(dict.fromkeys(files)),
        folders=list(dict.fromkeys(folders)),
        missing=missing,
    )


def _dependencies(
    manifest: ReleaseManifest, tables: list[EvaluationTableRecord]
) -> list[BundleRecord]:
    """Every bundle the tables were produced from, in manifest order."""
    bundles = {bundle.path: bundle for bundle in manifest.bundles}
    reached: set[str] = set()
    pending = [
        bundle_input.path for table in tables for bundle_input in table.inputs
    ]
    while pending:
        path = pending.pop()
        if path in reached:
            continue
        if path not in bundles:
            msg = (
                f"Release {manifest.release_id!r} names input bundle "
                f"{path!r} but lists no such bundle."
            )
            raise ValueError(msg)
        reached.add(path)
        pending.extend(
            bundle_input.path for bundle_input in bundles[path].inputs
        )
    return [bundle for bundle in manifest.bundles if bundle.path in reached]
