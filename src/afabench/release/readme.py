"""Human-readable release coverage, generated from the saved artifact index."""

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
    release_notes: str | None = None,
) -> str:
    """Describe actual coverage, not the workflow's requested combinations."""
    coverage = manifest.coverage
    lines = [
        f"# AFABench release {escape(manifest.release_id)}",
        "",
        f"Created: {escape(manifest.created_at)}  ",
        f"Execution mode: {manifest.execution_mode}",
        "",
    ]
    if manifest.smoke_artifacts:
        lines.extend(["Smoke-test outputs: not benchmark results.", ""])
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
            f"- Evaluations: {len(manifest.evaluations)}",
            f"- Included bundles: {len(manifest.bundles)}",
            "",
            "These lists summarize included artifacts; not every combination "
            "is necessarily present.",
            "",
            "## Omitted dataset bundles",
            "",
            "Evaluations and models for these datasets are included; obtain "
            "the datasets themselves locally.",
            "",
        ]
    )
    omitted = sorted(
        {
            (entry.bundle.dataset_key or "Unknown", entry.reason)
            for entry in manifest.omitted_dataset_bundles
        }
    )
    lines.extend(
        f"- {escape(dataset)}: {escape(reason)}" for dataset, reason in omitted
    )
    if not omitted:
        lines.append("- None")
    lines.extend(
        [
            "",
            "See the "
            + _link("full release manifest", "release_manifest.json")
            + " for exact evaluation identities and table paths.",
            "",
        ]
    )
    return "\n".join(lines)
