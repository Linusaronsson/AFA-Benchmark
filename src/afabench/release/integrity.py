"""Streaming byte inventory of a saved release package, excluding itself."""

import hashlib
from pathlib import Path

from afabench.release.manifest import RELEASE_MANIFEST_FILENAME


def package_file_inventory(package_dir: Path) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for path in sorted(package_dir.rglob("*")):
        if path.is_symlink():
            msg = f"Release packages cannot contain symlinks: {path}"
            raise ValueError(msg)
        if not path.is_file():
            continue
        relative = path.relative_to(package_dir).as_posix()
        # The inventory lives in the manifest, so it cannot hash itself.
        if relative == RELEASE_MANIFEST_FILENAME:
            continue
        with path.open("rb") as stream:
            inventory[relative] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return inventory
