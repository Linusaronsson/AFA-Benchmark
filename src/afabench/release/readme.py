"""Human-readable release coverage, generated from the saved artifact index."""

from collections import Counter
from collections.abc import Iterable
from html import escape
from urllib.parse import quote

from afabench.release.manifest import ReleaseManifest

RELEASE_README_FILENAME = "README.md"


def _values(values: Iterable[object]) -> str:
    return (
        ", ".join(sorted({escape(str(value)) for value in values})) or "None"
    )


def _link(label: str, path: str) -> str:
    return f"[{label}](./{quote(path, safe='/')})"


def render_release_readme(
    manifest: ReleaseManifest,
    *,
    output_paths: Iterable[str],
    release_notes: str | None = None,
) -> str:
    """Describe actual coverage, not the workflow's requested combinations."""
    coverage = manifest.coverage
    evaluations = manifest.evaluations
    lines = [
        f"# AFABench release {escape(manifest.release_id)}",
        "",
        f"Created: {escape(manifest.created_at)}  ",
        f"Execution mode: {manifest.execution_mode}",
        "",
    ]
    if release_notes and release_notes.strip():
        lines.extend(["## Release notes", "", release_notes.strip(), ""])
    lines.extend(
        [
            "## Coverage",
            "",
            f"- Datasets: {_values(coverage.datasets)}",
            f"- Methods: {_values(coverage.methods)}",
            "- Dataset realization indices: "
            f"{_values(coverage.dataset_realization_indices)}",
            f"- Evaluation splits: {_values(coverage.eval_splits)}",
            f"- Budget settings: {_values(coverage.budget_settings)}",
            f"- Classifier variants: {_values(coverage.classifier_variants)}",
            f"- Evaluations: {len(evaluations)}",
            f"- Included bundles: {len(manifest.bundles)}",
            "",
            "These lists summarize included artifacts; not every combination "
            "is necessarily present. Counts below are evaluation entries, "
            "not evaluated instances.",
            "",
            "### Evaluations by dataset and method",
            "",
            "| Dataset | Method | Evaluations |",
            "| --- | --- | ---: |",
        ]
    )
    pairs = Counter(
        (entry.dataset_key or "Unknown", entry.method_name or "Unknown")
        for entry in evaluations
    )
    for (dataset, method), count in sorted(pairs.items()):
        lines.append(
            f"| {escape(dataset).replace('|', '&#124;')} | "
            f"{escape(method).replace('|', '&#124;')} | {count} |"
        )
    if not evaluations:
        lines.extend(["", "No evaluation tables are included."])
    lines.extend(["", "### Budgets and seeds", ""])
    for label, field in [
        ("Training hard budgets", "train_hard_budget"),
        ("Evaluation hard budgets", "eval_hard_budget"),
        ("Training soft-budget parameters", "train_soft_budget_param"),
        ("Evaluation soft-budget parameters", "eval_soft_budget_param"),
        ("Training seeds", "train_seed"),
        ("Evaluation seeds", "eval_seed"),
    ]:
        lines.append(
            f"- {label}: "
            + _values(
                getattr(entry, field)
                for entry in evaluations
                if getattr(entry, field) is not None
            )
        )
    lines.extend(_available_files(manifest, output_paths))
    lines.extend(_caveats(manifest))
    lines.extend(
        [
            "",
            "See the "
            + _link("full release manifest", "release_manifest.json")
            + " for exact evaluation identities, table paths, producing "
            "commits and dataset redistribution reviews.",
            "",
            "Coverage is generated automatically from the release manifest.",
            "",
        ]
    )
    return "\n".join(lines)


def _available_files(
    manifest: ReleaseManifest, output_paths: Iterable[str]
) -> list[str]:
    paths = sorted(output_paths)
    evaluations = manifest.evaluations
    lines = [
        "",
        "## Available files",
        "",
        "- "
        + _link("Runnable benchmark config", "benchmark.yaml")
        + "; downloads set its output root to the destination. Run it with "
        + "`workflow/run.py --configfile <download-root>/benchmark.yaml`.",
    ]
    for label, field in [
        ("Raw evaluation tables", "raw_path"),
        ("Plotting-ready evaluation tables", "transformed_path"),
    ]:
        tables = [getattr(entry, field) for entry in evaluations]
        present = sorted({path for path in tables if path is not None})
        example = (
            "; " + _link("example table", "output/" + present[0])
            if present
            else ""
        )
        lines.append(f"- {label}: {len(present)}{example}")
    plots = [path for path in paths if path.startswith("plots/")]
    lines.append(f"- Plot files: {len(plots)}")
    lines.extend(
        "- " + _link(escape(category), f"output/{category}/")
        for category in sorted(
            {path.split("/")[0] for path in paths if "/" in path}
        )
    )
    if manifest.job_duration_table is not None:
        lines.append(
            "- "
            + _link("Job duration table", "release_job_duration_table.parquet")
        )
    return lines


def _caveats(manifest: ReleaseManifest) -> list[str]:
    lines = ["", "## Caveats", ""]
    if manifest.smoke_artifacts:
        lines.extend(["- Smoke-test outputs: not benchmark results."])
    for dataset, review in sorted(manifest.dataset_redistribution.items()):
        if review.notes:
            lines.append(f"- {escape(dataset)}: {escape(review.notes)}")
    for dataset, reason in sorted(
        {
            (entry.bundle.dataset_key or "Unknown", entry.reason)
            for entry in manifest.omitted_dataset_bundles
        }
    ):
        lines.append(
            f"- Dataset bundles omitted for {escape(dataset)}: {escape(reason)}"
        )
    return lines
