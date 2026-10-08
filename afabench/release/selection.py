"""
Choose which payloads of a benchmark release a download fetches.

Selection reads only the release manifest. Evaluations are matched
against the requested coverage; the bundles they depend on are found by
following the `inputs` of those evaluations and, in turn, of the bundles,
by content hash, then kept if their payload category is requested. Pretrained-model and
AFA-method bundles come with the time record their job wrote beside them.
So asking for evaluation tables alone fetches no bundle, and asking for
dataset and classifier bundles fetches the shared prerequisites of the
selected evaluations without their AFA-method bundles. Output categories
(top-level folders of the output root such as `plot_results`) are not
described per file by the manifest, so they are selected whole. The job
duration table is one file beside the manifest holding every job of the
release, so coverage does not narrow it either.

Requested coverage the release does not have, and inputs no bundle of the
release matches, are reported in `missing`, never filled from elsewhere.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from afabench.release.manifest import (
    INPUT_CATEGORIES,
    BudgetSetting,
    BundleEntry,
    ClassifierVariant,
    EvaluationEntry,
    IndexedInput,
    PayloadCategory,
    ReleaseManifest,
)


@dataclass(frozen=True, kw_only=True)
class ReleaseSelection:
    """
    What to download from one release.

    Empty coverage lists do not restrict; coverage never restricts output
    categories or the job duration table. A table matches a classifier
    variant if its raw table holds that variant's predictions, so a table
    without its raw table matches none.
    """

    payload_categories: list[PayloadCategory] = field(default_factory=list)
    output_categories: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    dataset_realization_indices: list[int] = field(default_factory=list)
    eval_splits: list[str] = field(default_factory=list)
    initializers: list[str] = field(default_factory=list)
    budget_settings: list[BudgetSetting] = field(default_factory=list)
    classifier_variants: list[ClassifierVariant] = field(default_factory=list)


@dataclass(frozen=True, kw_only=True)
class SelectedPayloads:
    """
    Paths relative to the release's output root.

    `job_duration_table` is whether to fetch the release's job duration
    table, beside its manifest.
    """

    files: list[str]
    folders: list[str]
    job_duration_table: bool
    missing: list[str]


type TableValues = Callable[[EvaluationEntry], Sequence[object]]


def select_payloads(
    manifest: ReleaseManifest, selection: ReleaseSelection
) -> SelectedPayloads:
    dimensions: list[tuple[str, Sequence[object], TableValues]] = [
        ("dataset", selection.datasets, lambda t: [t.dataset_key]),
        ("method", selection.methods, lambda t: [t.method_name]),
        (
            "dataset realization",
            selection.dataset_realization_indices,
            lambda t: [t.dataset_realization_index],
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
            lambda t: t.classifier_variants,
        ),
    ]
    tables = [
        table
        for table in manifest.evaluations
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

    for table in tables:
        for category, path in [
            (PayloadCategory.RAW_EVALUATION_TABLE, table.raw_path),
            (
                PayloadCategory.TRANSFORMED_EVALUATION_TABLE,
                table.transformed_path,
            ),
        ]:
            if category not in categories:
                continue
            if path is None:
                missing.append(
                    f"{category} of the evaluation {table.path}: "
                    "not in the release"
                )
            else:
                files.append(path)
    bundles, dangling = _dependencies(manifest, tables)
    folders.extend(
        _job_folder(bundle)
        for bundle in bundles
        if bundle.category in categories
    )
    missing.extend(
        f"{INPUT_CATEGORIES[bundle_input.role]} {bundle_input.path}, "
        f"{bundle_input.role} input of {artifact_path}: not in the release"
        for artifact_path, bundle_input in dangling
        if INPUT_CATEGORIES[bundle_input.role] in categories
    )
    for output_category in dict.fromkeys(selection.output_categories):
        if output_category in manifest.coverage.output_categories:
            folders.append(output_category)
        else:
            missing.append(
                f"output category {output_category}: not in the release"
            )
    job_duration_table = PayloadCategory.JOB_DURATION_TABLE in categories
    if job_duration_table and manifest.job_duration_table is None:
        missing.append(
            f"{PayloadCategory.JOB_DURATION_TABLE}: not in the release"
        )
        job_duration_table = False
    return SelectedPayloads(
        files=list(dict.fromkeys(files)),
        folders=list(dict.fromkeys(folders)),
        job_duration_table=job_duration_table,
        missing=missing,
    )


def _job_folder(bundle: BundleEntry) -> str:
    """
    Return the folder to fetch for a bundle: it, or its job's folder.

    The pretraining and training jobs write a time record beside their
    bundle, which the workflow's time aggregation reads. Without it the
    workflow reruns the job, replacing the restored bundle. Each of these
    bundles' parent folders holds the outputs of that one job only.
    """
    if bundle.category in {
        PayloadCategory.PRETRAINED_MODEL_BUNDLE,
        PayloadCategory.AFA_METHOD_BUNDLE,
    }:
        return str(PurePosixPath(bundle.path).parent)
    return bundle.path


def _dependencies(
    manifest: ReleaseManifest, tables: list[EvaluationEntry]
) -> tuple[list[BundleEntry], list[tuple[str, IndexedInput]]]:
    """
    Every bundle the tables were produced from, in manifest order.

    Also returns the inputs on the way that no bundle of the release
    matches by content hash, each with the path of the artifact naming it.
    """
    bundles: dict[str, list[BundleEntry]] = {}
    for bundle in manifest.bundles:
        if bundle.content_hash is not None:
            bundles.setdefault(bundle.content_hash, []).append(bundle)
    reached: set[str] = set()
    dangling: list[tuple[str, IndexedInput]] = []
    pending = [
        (table.path, bundle_input)
        for table in tables
        for bundle_input in table.inputs
    ]
    while pending:
        artifact_path, bundle_input = pending.pop()
        content_hash = bundle_input.content_hash
        if content_hash is None or content_hash not in bundles:
            if (artifact_path, bundle_input) not in dangling:
                dangling.append((artifact_path, bundle_input))
            continue
        for bundle in bundles[content_hash]:
            if bundle.path in reached:
                continue
            reached.add(bundle.path)
            pending.extend((bundle.path, entry) for entry in bundle.inputs)
    return (
        [bundle for bundle in manifest.bundles if bundle.path in reached],
        sorted(dangling, key=lambda item: (item[0], item[1].path)),
    )
