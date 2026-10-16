"""
Publish an output snapshot as a benchmark release and download it back.

The release host is reached only through a `ReleaseTransport`, a plain file
store addressed by POSIX repository paths; `afabench.release.huggingface`
adapts Hugging Face to it. A published release is the snapshot directory
uploaded file by file under `releases/<release_id>/`, so its Parquet tables
and plots stay ordinary downloadable files.

A download first resolves one release by its explicit id and keeps its
manifest. Everything it then fetches comes from that release's
folder, so a release published meanwhile cannot be mixed in. It fetches
either the whole release or the payloads `afabench.release.selection`
chose, into a staging directory, and restores them with
`restore_snapshot`. The host keeps file bytes only, so publishing adds
`output_mtimes.json`, the mtime of every file and directory under
`output/`, which downloading puts back before restoring: Snakemake judges
restored outputs by mtime.

A package holding any smoke-test artifact is never published
(`docs/adr/0012-a-downloaded-release-is-a-cache.md`); the test suite checks
the round trip by laying a smoke package out on a local host with
`release_files`. A release is refused while any dataset in its manifest's
`dataset_redistribution` is not `permitted` and its dataset bundles are
present. There is no redistribution override. Models and results remain
eligible regardless of their dataset's status. It is also refused while
any of its artifacts was produced from a
dirty tree or unknown code, unless the maintainer allows dirty code
(`docs/adr/0006-release-manifest-indexes-artifact-provenance.md`). The
smoke, redistribution and code checks live here rather than in a
transport, so no transport can skip them.
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
    JOB_DURATION_TABLE_FILENAME,
    RELEASE_MANIFEST_FILENAME,
    PayloadCategory,
    ReleaseManifest,
    code_by_stage,
    dataset_bundle_permitted,
    index_artifacts,
    read_release_manifest,
)
from afabench.release.selection import SelectedPayloads
from afabench.release.snapshot import SNAPSHOT_OUTPUT_SUBDIR, restore_snapshot

RELEASES_FOLDER = "releases"
OUTPUT_MTIMES_FILENAME = "output_mtimes.json"
# One path segment without glob characters, so a release cannot address
# another release's files or match more than its own folder on download.
RELEASE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# How many smoke-test artifacts a refused package names
SMOKE_ARTIFACTS_SHOWN = 5


class ReleaseTransport(Protocol):
    """A file store on the release host, addressed by repository paths."""

    def file_exists(self, path: str) -> bool: ...

    def upload_files(
        self, files: Mapping[str, Path | bytes], message: str
    ) -> None:
        """Upload every `{repository path: content}` in one commit."""
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
    allow_dirty_code: bool = False,
) -> ReleaseManifest:
    """
    Upload the snapshot in `package_dir` as its manifest's release.

    Dataset bundles require explicit redistribution permission, with no
    override. `allow_dirty_code` publishes artifacts produced from a
    dirty tree or unknown code anyway.
    """
    manifest = read_release_manifest(package_dir / RELEASE_MANIFEST_FILENAME)
    folder = release_folder(manifest.release_id)
    _check_smoke(manifest)
    _check_redistribution(package_dir, manifest)
    _check_code(manifest, allow_dirty_code=allow_dirty_code)
    message = f"Publish benchmark release {manifest.release_id}"
    # The host's history records what the maintainer allowed through.
    if allow_dirty_code:
        message += (
            "; artifacts produced from dirty or unknown code allowed by "
            "the maintainer"
        )
    # A published release is never replaced, so its identity keeps naming
    # the same outputs.
    if transport.file_exists(f"{folder}/{RELEASE_MANIFEST_FILENAME}"):
        msg = f"Release {manifest.release_id!r} is already published."
        raise FileExistsError(msg)
    transport.upload_files(release_files(package_dir), message=message)
    return manifest


def release_files(package_dir: Path) -> dict[str, Path | bytes]:
    """
    Lay out the snapshot in `package_dir` as its release on the host.

    Returns `{repository path: content}`, the files `publish_release`
    uploads after its checks. The test suite uploads them directly to
    check the round trip of a package `publish_release` refuses.
    """
    manifest = read_release_manifest(package_dir / RELEASE_MANIFEST_FILENAME)
    folder = release_folder(manifest.release_id)
    files: dict[str, Path | bytes] = {
        f"{folder}/{path.relative_to(package_dir).as_posix()}": path
        for path in sorted(package_dir.rglob("*"))
        if path.is_file()
    }
    files[f"{folder}/{OUTPUT_MTIMES_FILENAME}"] = _output_mtimes(
        package_dir / SNAPSHOT_OUTPUT_SUBDIR
    )
    return files


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
    release: str, transport: ReleaseTransport
) -> ResolvedRelease:
    """Resolve the explicitly named release and keep its manifest."""
    folder = release_folder(release)
    if not transport.file_exists(f"{folder}/{RELEASE_MANIFEST_FILENAME}"):
        msg = f"No release {release!r} is published."
        raise FileNotFoundError(msg)
    return _fetch_manifest(transport, folder, release)


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
    """Fetch only the selected payloads of `release`; restore them."""
    if not (payloads.files or payloads.folders or payloads.job_duration_table):
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
                *(
                    [f"{release.folder}/{JOB_DURATION_TABLE_FILENAME}"]
                    if payloads.job_duration_table
                    else []
                ),
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


def release_folder(release_id: str) -> str:
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        msg = (
            f"Release id {release_id!r} must be letters, digits, '.', '_' "
            "or '-', starting with a letter or digit."
        )
        raise ValueError(msg)
    return f"{RELEASES_FOLDER}/{release_id}"


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


def _check_smoke(manifest: ReleaseManifest) -> None:
    smoke = manifest.smoke_artifacts
    if not smoke:
        return
    shown = SMOKE_ARTIFACTS_SHOWN
    msg = (
        f"Release {manifest.release_id!r} holds {len(smoke)} smoke-test "
        f"artifact{'' if len(smoke) == 1 else 's'}, likely from a smoke test "
        "run into its output root, and smoke-test artifacts are never "
        "published. Remove them and save the snapshot again:\n  "
        + "\n  ".join(smoke[:shown])
        + (
            f"\n  ... and {len(smoke) - shown} more"
            if len(smoke) > shown
            else ""
        )
    )
    raise ValueError(msg)


def _check_redistribution(
    package_dir: Path, manifest: ReleaseManifest
) -> None:
    # Inspect the package, not just the saved index: a bundle added after
    # save must not silently bypass the redistribution gate.
    output_root = package_dir / SNAPSHOT_OUTPUT_SUBDIR
    misplaced = [
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*.bundle")
        if not path.is_relative_to(output_root)
    ]
    if misplaced:
        msg = (
            f"Release {manifest.release_id!r} holds bundles outside output/: "
            + ", ".join(misplaced)
        )
        raise ValueError(msg)
    index = index_artifacts(output_root)
    if index.unrecorded:
        msg = (
            f"Release {manifest.release_id!r} holds artifacts without "
            "provenance; save the package again:\n  "
            + "\n  ".join(index.unrecorded)
        )
        raise ValueError(msg)
    refused = [
        bundle
        for bundle in index.bundles
        if bundle.category is PayloadCategory.DATASET_BUNDLE
        and not dataset_bundle_permitted(
            bundle, manifest.dataset_redistribution
        )
    ]
    if refused:
        described = ", ".join(
            f"{bundle.dataset_key or 'unknown dataset'} ({bundle.path})"
            for bundle in refused
        )
        msg = (
            f"Release {manifest.release_id!r} holds dataset bundles without "
            f"permitted redistribution: {described}. Review them in "
            f"{DATASET_REDISTRIBUTION_FILE} and save the package again; "
            "unreviewed and restricted dataset bundles cannot be published."
        )
        raise ValueError(msg)
    if index.bundles != manifest.bundles:
        msg = (
            f"Release {manifest.release_id!r} bundle contents do not match "
            "its manifest; save the package again."
        )
        raise ValueError(msg)


def _check_code(manifest: ReleaseManifest, *, allow_dirty_code: bool) -> None:
    unclean = [
        stage_code
        for stage_code in code_by_stage(manifest)
        if not stage_code.code.clean
    ]
    if not unclean:
        if allow_dirty_code:
            msg = (
                f"Release {manifest.release_id!r} has no artifact produced "
                "from dirty or unknown code; no allowance is needed."
            )
            raise ValueError(msg)
        return
    if allow_dirty_code:
        return
    described = ", ".join(
        f"{stage_code.artifacts} {stage_code.stage} artifact(s) at "
        f"{stage_code.code.commit or 'unknown commit'}"
        + (" (dirty)" if stage_code.code.dirty else "")
        for stage_code in unclean
    )
    msg = (
        f"Release {manifest.release_id!r} holds artifacts produced from "
        f"dirty or unknown code: {described}. Regenerate them from a clean "
        "commit and save the snapshot again, or allow dirty code."
    )
    raise ValueError(msg)
