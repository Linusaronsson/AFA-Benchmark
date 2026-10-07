"""
Publish an output snapshot as a benchmark release and download it back.

The release host is reached only through a `ReleaseTransport`, a plain file
store addressed by POSIX repository paths; `afabench.release.huggingface`
adapts Hugging Face to it. A published release is the snapshot directory
uploaded file by file under `releases/<release_id>/`, so its Parquet tables
and plots stay ordinary downloadable files.

A download first resolves one release, by id or as the latest `full` one
(by manifest `created_at`; partial releases are never the latest), and
keeps its manifest. Everything it then fetches comes from that release's
folder, so a release published meanwhile cannot be mixed in. It fetches
either the whole release or the payloads `afabench.release.selection`
chose, into a staging directory, and restores them with
`restore_snapshot`. The host keeps file bytes only, so publishing adds
`output_mtimes.json`, the mtime of every file and directory under
`output/`, which downloading puts back before restoring: Snakemake judges
restored outputs by mtime.

Test-only packages (smoke outputs, see `ReleaseScope`) are only ever
published under `test_releases/<release_id>/`, so a maintainer can check
the host round trip without promoting them to official releases. An
official release is refused while any dataset in its manifest's
`settings.dataset_redistribution` is not `permitted`, unless the maintainer
allows that dataset by name. The scope and redistribution checks live here
rather than in a transport, so no transport can skip them.
"""

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from afabench.release.manifest import (
    DATASET_REDISTRIBUTION_FILE,
    RELEASE_MANIFEST_FILENAME,
    RedistributionStatus,
    ReleaseManifest,
    ReleaseScope,
    read_release_manifest,
)
from afabench.release.selection import SelectedPayloads
from afabench.release.snapshot import SNAPSHOT_OUTPUT_SUBDIR, restore_snapshot

RELEASES_FOLDER = "releases"
TEST_RELEASES_FOLDER = "test_releases"
OUTPUT_MTIMES_FILENAME = "output_mtimes.json"
# One path segment without glob characters, so a release cannot address
# another release's files or match more than its own folder on download.
RELEASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# Selects the latest full release, so no release can be named this.
LATEST_RELEASE = "latest"


class ReleaseTransport(Protocol):
    """A file store on the release host, addressed by repository paths."""

    def file_exists(self, path: str) -> bool: ...

    def upload_files(
        self, files: Mapping[str, Path | bytes], message: str
    ) -> None:
        """Upload every `{repository path: content}` in one commit."""
        ...

    def list_folders(self, folder: str) -> list[str]:
        """Return the names of the folders directly under `folder`."""
        ...

    def download_files(self, paths: Sequence[str], local_dir: Path) -> None:
        """Write each file in `paths` to `local_dir/<repository path>`."""
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
    allow_redistribution: Sequence[str] = (),
) -> ReleaseManifest:
    """
    Upload the snapshot in `package_dir` as its manifest's release.

    `allow_redistribution` names the unreviewed or restricted dataset keys
    a maintainer publishes in an official release anyway.
    """
    manifest = read_release_manifest(package_dir / RELEASE_MANIFEST_FILENAME)
    _check_scope(manifest, test_release=test_release)
    folder = release_folder(manifest.release_id, test_release=test_release)
    message = f"Publish benchmark release {manifest.release_id}"
    if test_release:
        if allow_redistribution:
            msg = (
                "Dataset redistribution is reviewed only for official "
                "releases; a test release needs no allowance."
            )
            raise ValueError(msg)
    else:
        _check_redistribution(manifest, allow_redistribution)
        if allow_redistribution:
            # The host's history records which datasets were allowed through.
            message += (
                "; redistribution allowed by the maintainer for unreviewed "
                "or restricted datasets: "
                + ", ".join(sorted(allow_redistribution))
            )
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
    transport.upload_files(files, message=message)
    return manifest


@dataclass(frozen=True, kw_only=True)
class ResolvedRelease:
    """
    One published release, fixed before anything is downloaded from it.

    `manifest_bytes` is the manifest as published; the download restores
    these bytes instead of fetching the manifest again.
    """

    folder: str
    manifest: ReleaseManifest
    manifest_bytes: bytes


def resolve_release(
    release: str, transport: ReleaseTransport, *, test_release: bool = False
) -> ResolvedRelease:
    """Fix the release `release` names: a release id or `LATEST_RELEASE`."""
    if release == LATEST_RELEASE:
        if test_release:
            msg = (
                f"{LATEST_RELEASE!r} selects the latest {ReleaseScope.FULL} "
                "release; name a test release by its release id."
            )
            raise ValueError(msg)
        return _latest_full_release(transport)
    folder = release_folder(release, test_release=test_release)
    if not transport.file_exists(f"{folder}/{RELEASE_MANIFEST_FILENAME}"):
        kind = "test release" if test_release else "official release"
        msg = f"No {kind} {release!r} is published."
        raise FileNotFoundError(msg)
    resolved = _fetch_manifest(transport, folder, release)
    _check_scope(resolved.manifest, test_release=test_release)
    return resolved


def download_release(
    release: ResolvedRelease,
    transport: ReleaseTransport,
    destination_root: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Fetch every file of `release` and restore it."""
    with tempfile.TemporaryDirectory() as staging:
        transport.download_folder(release.folder, Path(staging))
        package_dir = Path(staging) / release.folder
        # A published release is never replaced, so this guards against a
        # release edited on the host by other means.
        manifest_path = package_dir / RELEASE_MANIFEST_FILENAME
        if manifest_path.read_bytes() != release.manifest_bytes:
            msg = (
                f"The manifest of release {release.manifest.release_id!r} "
                "changed on the host during the download."
            )
            raise ValueError(msg)
        _set_output_mtimes(package_dir)
        restore_snapshot(package_dir, destination_root, overwrite=overwrite)


def download_selection(
    release: ResolvedRelease,
    payloads: SelectedPayloads,
    transport: ReleaseTransport,
    destination_root: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Fetch only the selected files and folders of `release`; restore them."""
    if not payloads.files and not payloads.folders:
        msg = (
            "Nothing selected is in release "
            f"{release.manifest.release_id!r}:\n" + "\n".join(payloads.missing)
        )
        raise LookupError(msg)
    output = f"{release.folder}/{SNAPSHOT_OUTPUT_SUBDIR}"
    with tempfile.TemporaryDirectory() as staging:
        transport.download_files(
            [
                f"{release.folder}/{OUTPUT_MTIMES_FILENAME}",
                *(f"{output}/{path}" for path in payloads.files),
            ],
            Path(staging),
        )
        for folder in payloads.folders:
            transport.download_folder(f"{output}/{folder}", Path(staging))
        package_dir = Path(staging) / release.folder
        (package_dir / RELEASE_MANIFEST_FILENAME).write_bytes(
            release.manifest_bytes
        )
        _set_output_mtimes(package_dir, selected_only=True)
        restore_snapshot(package_dir, destination_root, overwrite=overwrite)


def _latest_full_release(transport: ReleaseTransport) -> ResolvedRelease:
    releases = [
        _fetch_manifest(transport, release_folder(release_id), release_id)
        for release_id in transport.list_folders(RELEASES_FOLDER)
    ]
    full = [
        release
        for release in releases
        if release.manifest.scope is ReleaseScope.FULL
    ]
    if not full:
        others = ", ".join(
            f"{release.manifest.release_id} ({release.manifest.scope})"
            for release in releases
        )
        msg = (
            f"No {ReleaseScope.FULL} release is published, so there is no "
            "latest one; name a release id to download it. Published: "
            f"{others or 'none'}."
        )
        raise LookupError(msg)
    return max(
        full,
        key=lambda release: (
            release.manifest.created_at,
            release.manifest.release_id,
        ),
    )


def _fetch_manifest(
    transport: ReleaseTransport, folder: str, release_id: str
) -> ResolvedRelease:
    path = f"{folder}/{RELEASE_MANIFEST_FILENAME}"
    with tempfile.TemporaryDirectory() as staging:
        transport.download_files([path], Path(staging))
        manifest_path = Path(staging) / path
        manifest = read_release_manifest(manifest_path)
        manifest_bytes = manifest_path.read_bytes()
    # Guards against a release edited on the host by other means.
    if manifest.release_id != release_id:
        msg = (
            f"Release {release_id!r} holds a manifest for release "
            f"{manifest.release_id!r}."
        )
        raise ValueError(msg)
    return ResolvedRelease(
        folder=folder, manifest=manifest, manifest_bytes=manifest_bytes
    )


def release_folder(release_id: str, *, test_release: bool = False) -> str:
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        msg = (
            f"Release id {release_id!r} must be letters, digits, '.', '_' "
            "or '-', starting with a letter or digit."
        )
        raise ValueError(msg)
    if release_id == LATEST_RELEASE:
        msg = f"Release id {release_id!r} is reserved for the latest release."
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


def _set_output_mtimes(
    package_dir: Path, *, selected_only: bool = False
) -> None:
    output_root = package_dir / SNAPSHOT_OUTPUT_SUBDIR
    mtimes: dict[str, dict[str, int]] = json.loads(
        (package_dir / OUTPUT_MTIMES_FILENAME).read_text()
    )
    if selected_only:
        # Only what was downloaded is restored; no other directory of the
        # release is created.
        mtimes = {
            kind: {
                relative: mtime_ns
                for relative, mtime_ns in entries.items()
                if (output_root / relative).exists()
            }
            for kind, entries in mtimes.items()
        }
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


def _check_redistribution(
    manifest: ReleaseManifest, allow_redistribution: Sequence[str]
) -> None:
    unresolved = {
        dataset: review.status
        for dataset, review in manifest.settings.dataset_redistribution.items()
        if review.status is not RedistributionStatus.PERMITTED
    }
    needless = [
        dataset
        for dataset in allow_redistribution
        if dataset not in unresolved
    ]
    if needless:
        msg = (
            f"Release {manifest.release_id!r} has no unreviewed or "
            f"restricted dataset {', '.join(needless)}; only those can be "
            "allowed."
        )
        raise ValueError(msg)
    refused = [
        f"{dataset} ({status})"
        for dataset, status in unresolved.items()
        if dataset not in allow_redistribution
    ]
    if refused:
        msg = (
            f"Release {manifest.release_id!r} holds datasets whose "
            f"redistribution is not {RedistributionStatus.PERMITTED}: "
            f"{', '.join(refused)}. Review them in "
            f"{DATASET_REDISTRIBUTION_FILE} and save the snapshot again, or "
            "allow each by name."
        )
        raise ValueError(msg)
