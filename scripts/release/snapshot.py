"""Save and restore an output snapshot (see `afabench.release.snapshot`)."""

from pathlib import Path
from typing import Final

import typer

from afabench.release.snapshot import restore_snapshot, save_snapshot

app = typer.Typer()
DEFAULT_OUTPUT_ROOT: Final[Path] = Path("extra/output")


@app.command()
def save(
    snapshot_dir: Path,
    source_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    overwrite: bool = False,
) -> None:
    save_snapshot(source_root, snapshot_dir, overwrite=overwrite)


@app.command()
def restore(
    snapshot_dir: Path,
    destination_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    overwrite: bool = False,
) -> None:
    restore_snapshot(snapshot_dir, destination_root, overwrite=overwrite)


if __name__ == "__main__":
    app()
