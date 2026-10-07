"""
Save and restore a verbatim copy of the pipeline's output root.

A snapshot may carry a release manifest beside its `output/` tree
(`afabench.release.manifest`). Restore puts it beside the restored root, so
the snapshot layout is mirrored: `<root>/../release_manifest.json`.
"""

import shutil
from pathlib import Path

from afabench.release.manifest import (
    RELEASE_MANIFEST_FILENAME,
    ReleaseManifest,
    write_release_manifest,
)

SNAPSHOT_OUTPUT_SUBDIR = "output"


def save_snapshot(
    source_root: Path,
    snapshot_dir: Path,
    *,
    overwrite: bool = False,
    manifest: ReleaseManifest | None = None,
) -> None:
    """Copy every file under `source_root` into `snapshot_dir/output`."""
    manifest_path = snapshot_dir / RELEASE_MANIFEST_FILENAME
    if manifest is not None:
        _refuse_existing(manifest_path, overwrite=overwrite)
    _verbatim_copy(
        source_root,
        snapshot_dir / SNAPSHOT_OUTPUT_SUBDIR,
        overwrite=overwrite,
    )
    if manifest is not None:
        write_release_manifest(manifest, manifest_path)


def restore_snapshot(
    snapshot_dir: Path, destination_root: Path, *, overwrite: bool = False
) -> None:
    """Copy every file from `snapshot_dir/output` into `destination_root`."""
    manifest_path = snapshot_dir / RELEASE_MANIFEST_FILENAME
    restored_manifest_path = restored_manifest_location(destination_root)
    if manifest_path.is_file():
        _refuse_existing(restored_manifest_path, overwrite=overwrite)
    _verbatim_copy(
        snapshot_dir / SNAPSHOT_OUTPUT_SUBDIR,
        destination_root,
        overwrite=overwrite,
    )
    if manifest_path.is_file():
        shutil.copy2(manifest_path, restored_manifest_path)


def restored_manifest_location(destination_root: Path) -> Path:
    return destination_root.parent / RELEASE_MANIFEST_FILENAME


def _refuse_existing(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        msg = f"Refusing to overwrite existing release manifest: {path}"
        raise FileExistsError(msg)


def _verbatim_copy(
    source: Path, destination: Path, *, overwrite: bool
) -> None:
    if not source.is_dir():
        msg = f"Source root does not exist: {source}"
        raise FileNotFoundError(msg)
    files = [path for path in source.rglob("*") if path.is_file()]
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

    for source_file, target_file in targets:
        target_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target_file)

    # Snakemake falls back to a directory's own mtime for directory outputs
    # with no `.snakemake_timestamp`, so that mtime must round-trip too.
    for source_dir in [source, *(p for p in source.rglob("*") if p.is_dir())]:
        target_dir = destination / source_dir.relative_to(source)
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copystat(source_dir, target_dir)
