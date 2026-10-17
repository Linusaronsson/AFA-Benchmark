"""
Content-checked dataset edges for explicitly reused release artifacts.

Only existing artifacts still identified by the restored release are eligible.
Missing artifacts retain ordinary dependencies. Dataset generation must finish
in a separate invocation, so the checked bytes cannot be replaced mid-plan.
"""

from pathlib import Path
from typing import final

from afabench.core.bundle_system.bundle import (
    compute_content_hash,
    read_manifest,
)
from afabench.core.output_layout import OutputLayout
from afabench.evaluation.provenance import evaluation_table_provenance
from afabench.release.manifest import read_release_manifest


@final
class ReleaseReuse:
    def __init__(self, output_root: Path, *, preview: bool = False) -> None:
        self.output_root = output_root
        self.preview = preview
        self._dataset_hashes: dict[Path, str] = {}
        manifest = read_release_manifest(output_root / "release_manifest.json")
        self.bundles = {entry.path: entry for entry in manifest.bundles}
        self.evaluations = {
            entry.raw_path: entry
            for entry in manifest.evaluations
            if entry.raw_path is not None
        }

    def validate_datasets(
        self,
        layout: OutputLayout,
        datasets: list[str],
        realizations: list[int],
    ) -> None:
        # Snakemake can prune upstream jobs behind already-present targets.
        # Validate even those edges before it decides there is no work to do.
        for entry in [*self.bundles.values(), *self.evaluations.values()]:
            if (
                entry.dataset_key not in datasets
                or entry.dataset_realization_index not in realizations
            ):
                continue
            consumer = self.output_root / entry.path
            for item in entry.inputs:
                if item.role not in {
                    "train_dataset",
                    "val_dataset",
                    "eval_dataset",
                }:
                    continue
                split = (
                    "train"
                    if item.role == "train_dataset"
                    else "val"
                    if item.role == "val_dataset"
                    else getattr(entry, "eval_split", None)
                )
                if split is None:
                    message = f"Unknown evaluation split for {consumer}."
                    raise ValueError(message)
                dataset = Path(
                    layout.dataset_bundle(
                        dataset=entry.dataset_key,
                        dataset_realization_index=entry.dataset_realization_index,
                        split=split,
                    )
                )
                self.compatible(consumer, dataset, item.role)

    def _dataset_hash(self, dataset: Path) -> str | None:
        if not (dataset / "manifest.json").is_file():
            if self.preview:
                return None
            message = (
                f"Release reuse requires dataset bundle {dataset}; run "
                "all_generate_datasets first without reuse_release."
            )
            raise ValueError(message)
        if dataset not in self._dataset_hashes:
            self._dataset_hashes[dataset] = compute_content_hash(
                dataset / "data"
            )
        return self._dataset_hashes[dataset]

    def compatible(self, consumer: Path, dataset: Path, role: str) -> bool:
        relative = consumer.relative_to(self.output_root).as_posix()
        if not consumer.exists():
            return False
        bundle = self.bundles.get(relative)
        evaluation = self.evaluations.get(relative)
        entry = bundle or evaluation
        if entry is None:
            return False
        if bundle is not None and (
            read_manifest(consumer).get("content_hash") != bundle.content_hash
        ):
            # A locally replaced bundle is no longer the released artifact.
            return False
        if evaluation is not None:
            record = evaluation_table_provenance(consumer)
            if record is None or [
                (item.role, item.content_hash) for item in record.inputs
            ] != [
                (item.role, item.content_hash) for item in evaluation.inputs
            ]:
                return False
        expected = [
            item.content_hash for item in entry.inputs if item.role == role
        ]
        actual = self._dataset_hash(dataset)
        if (
            len(expected) != 1
            or expected[0] is None
            or (actual is not None and expected[0] != actual)
        ):
            message = (
                f"Incompatible dataset {dataset} for released artifact "
                f"{consumer} ({role}): expected {expected}, got {actual}. "
                "Use a separate output root to train on changed data."
            )
            raise ValueError(message)
        return True
