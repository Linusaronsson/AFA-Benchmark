"""
Save and restore the pipeline's output root.

Local snapshots are verbatim; release packages omit dataset bundles excluded
by inclusion policy or redistribution permission, matching their manifest.

A snapshot may carry a release manifest beside its `output/` tree
(`afabench.release.manifest`), and the release's job duration table beside
that. Restore puts them inside the restored root instead,
`<root>/release_manifest.json`, so a checkout's smoke and production output
roots each keep their own. Save leaves them out of the copied root: a new
release gets its own manifest, never the one restored before.
"""

import shutil
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import yaml

from afabench.release.integrity import package_file_inventory
from afabench.release.manifest import (
    JOB_DURATION_TABLE_FILENAME,
    RELEASE_MANIFEST_FILENAME,
    ReleaseManifest,
    read_release_manifest,
    write_release_manifest,
)
from afabench.release.readme import (
    RELEASE_README_FILENAME,
    render_release_readme,
)
from afabench.release.workflow_config import BENCHMARK_CONFIG_FILENAME

SNAPSHOT_OUTPUT_SUBDIR = "output"
RELEASE_FILENAMES = (
    RELEASE_MANIFEST_FILENAME,
    JOB_DURATION_TABLE_FILENAME,
    RELEASE_README_FILENAME,
    BENCHMARK_CONFIG_FILENAME,
)


def save_snapshot(
    source_root: Path,
    snapshot_dir: Path,
    *,
    overwrite: bool = False,
    manifest: ReleaseManifest | None = None,
    job_duration_table: bytes | None = None,
    release_notes: str | None = None,
    on_copy_progress: Callable[[int, int], None] | None = None,
) -> None:
    """
    Copy `source_root` into `snapshot_dir/output`.

    With a release manifest, omit bundles absent from its index (dataset
    bundles without permission). `job_duration_table` describes the release.
    `on_copy_progress` receives (completed files, total files) before copying
    starts and after each successful file copy. Completion does not include
    directory timestamp preservation or writing the release files.
    """
    if release_notes is not None and manifest is None:
        msg = "Release notes require a release manifest."
        raise ValueError(msg)
    manifest_path = snapshot_dir / RELEASE_MANIFEST_FILENAME
    table_path = snapshot_dir / JOB_DURATION_TABLE_FILENAME
    if manifest is not None:
        for name in RELEASE_FILENAMES:
            _refuse_existing(snapshot_dir / name, overwrite=overwrite)
    skip = [
        source_root / name
        for name in RELEASE_FILENAMES
        if manifest is not None or name != BENCHMARK_CONFIG_FILENAME
    ]
    if manifest is not None:
        included = {bundle.path for bundle in manifest.bundles}
        # Include destination-only bundles too: --overwrite must not keep
        # an earlier release's unindexed data just because it left the source.
        bundle_paths = {
            path.relative_to(root)
            for root in [source_root, snapshot_dir / SNAPSHOT_OUTPUT_SUBDIR]
            for path in root.rglob("*.bundle")
            if path.is_dir()
        }
        skip.extend(
            source_root / path
            for path in sorted(bundle_paths)
            if path.as_posix() not in included
        )
    _verbatim_copy(
        source_root,
        snapshot_dir / SNAPSHOT_OUTPUT_SUBDIR,
        overwrite=overwrite,
        skip=skip,
        on_copy_progress=on_copy_progress,
    )
    if manifest is not None:
        if job_duration_table is None:
            # An overwritten release must not keep an earlier one's table.
            table_path.unlink(missing_ok=True)
        else:
            table_path.write_bytes(job_duration_table)
        (snapshot_dir / BENCHMARK_CONFIG_FILENAME).write_text(
            yaml.safe_dump(manifest.workflow_config.benchmark, sort_keys=False)
        )
        (snapshot_dir / RELEASE_README_FILENAME).write_text(
            render_release_readme(manifest, release_notes=release_notes),
            encoding="utf-8",
        )
        manifest = replace(
            manifest, file_inventory=package_file_inventory(snapshot_dir)
        )
        write_release_manifest(manifest, manifest_path)


def restore_snapshot(
    snapshot_dir: Path, destination_root: Path, *, overwrite: bool = False
) -> None:
    """
    Copy every file from `snapshot_dir/output` into `destination_root`.

    The release manifest and job duration table, if any, go inside it.
    """
    manifest_path = snapshot_dir / RELEASE_MANIFEST_FILENAME
    benchmark = None
    if manifest_path.is_file():
        benchmark = {
            **read_release_manifest(manifest_path).workflow_config.benchmark,
            "output_root": str(destination_root.resolve()),
        }
        _refuse_existing(
            destination_root / BENCHMARK_CONFIG_FILENAME,
            overwrite=overwrite,
        )
    release_files = [
        (snapshot_dir / name, destination_root / name)
        for name in RELEASE_FILENAMES
        if name != BENCHMARK_CONFIG_FILENAME
        and (snapshot_dir / name).is_file()
    ]
    for _, restored in release_files:
        _refuse_existing(restored, overwrite=overwrite)
    output = snapshot_dir / SNAPSHOT_OUTPUT_SUBDIR
    table = snapshot_dir / JOB_DURATION_TABLE_FILENAME
    # A download of the job duration table alone fetches no output tree.
    if output.exists() or not table.is_file():
        _verbatim_copy(output, destination_root, overwrite=overwrite)
    for path, restored in release_files:
        restored.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, restored)
    if benchmark is not None:
        (destination_root / BENCHMARK_CONFIG_FILENAME).write_text(
            yaml.safe_dump(benchmark, sort_keys=False)
        )


def restored_manifest_location(destination_root: Path) -> Path:
    return destination_root / RELEASE_MANIFEST_FILENAME


def _refuse_existing(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        msg = f"Refusing to overwrite existing release file: {path}"
        raise FileExistsError(msg)


def _remove_omitted(
    source: Path,
    destination: Path,
    skip: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    # Overwriting a release must not retain a dataset bundle from an
    # earlier save whose dataset release policy has since changed.
    for omitted in skip:
        if omitted.suffix != ".bundle":
            continue
        target = destination / omitted.relative_to(source)
        if target.exists():
            _refuse_existing(target, overwrite=overwrite)
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()


def _verbatim_copy(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
    skip: Sequence[Path] = (),
    on_copy_progress: Callable[[int, int], None] | None = None,
) -> None:
    if not source.is_dir():
        msg = f"Source root does not exist: {source}"
        raise FileNotFoundError(msg)

    def skipped(path: Path) -> bool:
        return any(path.is_relative_to(omitted) for omitted in skip)

    files = [
        path
        for path in source.rglob("*")
        if path.is_file() and not skipped(path)
    ]
    if not files:
        msg = f"Source root is empty: {source}"
        raise FileNotFoundError(msg)

    targets = [
        (path, destination / path.relative_to(source)) for path in files
    ]
    if not overwrite:
        conflicts = [target for _, target in targets if target.exists()]
        if conflicts:
            msg = (
                f"Refusing to overwrite existing path(s) in {destination}: "
                + ", ".join(str(path) for path in conflicts)
            )
            raise FileExistsError(msg)

    _remove_omitted(source, destination, skip, overwrite=overwrite)

    report_progress: Callable[[int, int], None] = on_copy_progress or (
        lambda _completed, _total: None
    )
    total = len(targets)
    report_progress(0, total)
    for completed, (source_file, target_file) in enumerate(targets, start=1):
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)
        report_progress(completed, total)

    # Snakemake falls back to a directory's own mtime for directory outputs
    # with no `.snakemake_timestamp`, so that mtime must round-trip too.
    # Every directory exists before any mtime is set, since creating an
    # empty one would update its parent's mtime.
    directories = [
        (source_dir, destination / source_dir.relative_to(source))
        for source_dir in [
            source,
            *(p for p in source.rglob("*") if p.is_dir() and not skipped(p)),
        ]
    ]
    for _, target_dir in directories:
        target_dir.mkdir(parents=True, exist_ok=True)
    for source_dir, target_dir in directories:
        shutil.copystat(source_dir, target_dir)
