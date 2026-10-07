"""Save and restore a verbatim copy of the pipeline's output root."""

import shutil
from pathlib import Path

SNAPSHOT_OUTPUT_SUBDIR = "output"


def save_snapshot(
    source_root: Path, snapshot_dir: Path, *, overwrite: bool = False
) -> None:
    """Copy every file under `source_root` into `snapshot_dir/output`."""
    _verbatim_copy(
        source_root,
        snapshot_dir / SNAPSHOT_OUTPUT_SUBDIR,
        overwrite=overwrite,
    )


def restore_snapshot(
    snapshot_dir: Path, destination_root: Path, *, overwrite: bool = False
) -> None:
    """Copy every file from `snapshot_dir/output` into `destination_root`."""
    _verbatim_copy(
        snapshot_dir / SNAPSHOT_OUTPUT_SUBDIR,
        destination_root,
        overwrite=overwrite,
    )


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
