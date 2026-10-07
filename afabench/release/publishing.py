"""
Publish an output snapshot as a benchmark release and download it back.

The release host is reached only through a `ReleaseTransport`, a plain file
store addressed by POSIX repository paths; `afabench.release.huggingface`
adapts Hugging Face to it. A published release is the snapshot directory
uploaded file by file under `releases/<release_id>/`, so its Parquet tables
and plots stay ordinary downloadable files. Downloading fetches one release
into a staging directory and restores it with `restore_snapshot`. The host
keeps file bytes only, so publishing adds `output_mtimes.json`, the mtime of
every file and directory under `output/`, which downloading puts back
before restoring: Snakemake judges restored outputs by mtime.

Test-only packages (smoke outputs, see `ReleaseScope`) are only ever
published under `test_releases/<release_id>/`, so a maintainer can check
the host round trip without promoting them to official releases. The scope
check lives here rather than in a transport, so no transport can skip it.
"""

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from afabench.release.manifest import (
    RELEASE_MANIFEST_FILENAME,
    ReleaseManifest,
    ReleaseScope,
    read_release_manifest,
)
from afabench.release.snapshot import SNAPSHOT_OUTPUT_SUBDIR, restore_snapshot

RELEASES_FOLDER = "releases"
TEST_RELEASES_FOLDER = "test_releases"
OUTPUT_MTIMES_FILENAME = "output_mtimes.json"
# One path segment without glob characters, so a release cannot address
# another release's files or match more than its own folder on download.
RELEASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ReleaseTransport(Protocol):
    """A file store on the release host, addressed by repository paths."""

    def file_exists(self, path: str) -> bool: ...

    def upload_files(
        self, files: Mapping[str, Path | bytes], message: str
    ) -> None:
        """Upload every `{repository path: content}` in one commit."""
        ...

    def download_folder(self, folder: str, local_dir: Path) -> None:
        """Write every file under `folder` to `local_dir/<repository path>`."""
        ...

    def folder_url(self, folder: str) -> str:
        """Return a public web page listing the files under `folder`."""
        ...


def publish_release(
    package_dir: Path,
    transport: ReleaseTransport,
    *,
    test_release: bool = False,
) -> ReleaseManifest:
    """Upload the snapshot in `package_dir` as its manifest's release."""
    manifest = read_release_manifest(package_dir / RELEASE_MANIFEST_FILENAME)
    _check_scope(manifest, test_release=test_release)
    folder = release_folder(manifest.release_id, test_release=test_release)
    # A published release is never replaced, so its identity keeps naming
    # the same outputs.
    if transport.file_exists(f"{folder}/{RELEASE_MANIFEST_FILENAME}"):
        msg = f"Release {manifest.release_id!r} is already published."
        raise FileExistsError(msg)
    files: dict[str, Path | bytes] = {
        f"{folder}/{path.relative_to(package_dir).as_posix()}": path
        for path in sorted(package_dir.rglob("*"))
        if path.is_file()
    }
    files[f"{folder}/{OUTPUT_MTIMES_FILENAME}"] = _output_mtimes(
        package_dir / SNAPSHOT_OUTPUT_SUBDIR
    )
    transport.upload_files(
        files, message=f"Publish benchmark release {manifest.release_id}"
    )
    return manifest


def download_release(
    release_id: str,
    transport: ReleaseTransport,
    destination_root: Path,
    *,
    overwrite: bool = False,
    test_release: bool = False,
) -> ReleaseManifest:
    """Fetch one release and restore it into `destination_root`."""
    folder = release_folder(release_id, test_release=test_release)
    if not transport.file_exists(f"{folder}/{RELEASE_MANIFEST_FILENAME}"):
        kind = "test release" if test_release else "official release"
        msg = f"No {kind} {release_id!r} is published."
        raise FileNotFoundError(msg)
    with tempfile.TemporaryDirectory() as staging:
        transport.download_folder(folder, Path(staging))
        package_dir = Path(staging) / folder
        manifest = read_release_manifest(
            package_dir / RELEASE_MANIFEST_FILENAME
        )
        # Guards against a release edited on the host by other means.
        if manifest.release_id != release_id:
            msg = (
                f"Release {release_id!r} holds a manifest for release "
                f"{manifest.release_id!r}."
            )
            raise ValueError(msg)
        _check_scope(manifest, test_release=test_release)
        _set_output_mtimes(package_dir)
        restore_snapshot(package_dir, destination_root, overwrite=overwrite)
    return manifest


def release_folder(release_id: str, *, test_release: bool = False) -> str:
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        msg = (
            f"Release id {release_id!r} must be letters, digits, '.', '_' "
            "or '-', starting with a letter or digit."
        )
        raise ValueError(msg)
    parent = TEST_RELEASES_FOLDER if test_release else RELEASES_FOLDER
    return f"{parent}/{release_id}"


def _output_mtimes(output_root: Path) -> bytes:
    paths = [output_root, *sorted(output_root.rglob("*"))]

    def relative(path: Path) -> str:
        return path.relative_to(output_root).as_posix()

    mtimes = {
        "files": {
            relative(path): path.stat().st_mtime_ns
            for path in paths
            if path.is_file()
        },
        "directories": {
            relative(path): path.stat().st_mtime_ns
            for path in paths
            if path.is_dir()
        },
    }
    return json.dumps(mtimes, indent=2).encode()


def _set_output_mtimes(package_dir: Path) -> None:
    output_root = package_dir / SNAPSHOT_OUTPUT_SUBDIR
    mtimes = json.loads((package_dir / OUTPUT_MTIMES_FILENAME).read_text())
    # The host drops empty directories. Create them all before setting any
    # mtime, since creating one updates its parent's mtime.
    for relative in mtimes["directories"]:
        (output_root / relative).mkdir(parents=True, exist_ok=True)
    for kind in ["files", "directories"]:
        for relative, mtime_ns in mtimes[kind].items():
            os.utime(output_root / relative, ns=(mtime_ns, mtime_ns))


def _check_scope(manifest: ReleaseManifest, *, test_release: bool) -> None:
    is_test_only = manifest.scope is ReleaseScope.TEST_ONLY
    if is_test_only and not test_release:
        msg = (
            f"Release {manifest.release_id!r} has scope "
            f"{ReleaseScope.TEST_ONLY}; it can only be a test release, "
            "never an official one."
        )
        raise ValueError(msg)
    if test_release and not is_test_only:
        msg = (
            f"Release {manifest.release_id!r} has scope {manifest.scope}; "
            f"only {ReleaseScope.TEST_ONLY} packages are test releases."
        )
        raise ValueError(msg)
