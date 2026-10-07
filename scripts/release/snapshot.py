"""
Save and restore an output snapshot (see `afabench.release.snapshot`).

`save --release-id` also writes a release manifest from the checkout and the
workflow configuration given by `--profile`, `--configfile` and `--config`
(see `docs/release_manifest.md`).

`publish` uploads such a snapshot to the release host and `download`
retrieves one release from it (see `afabench.release.publishing` and
`docs/release_publishing.md`). Nothing else here touches the host.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Final

import typer

from afabench.release.huggingface import HuggingFaceTransport
from afabench.release.manifest import (
    RELEASE_MANIFEST_FILENAME,
    ReleaseManifest,
    ReleaseScope,
    build_release_manifest,
    read_release_manifest,
)
from afabench.release.publishing import (
    ReleaseTransport,
    download_release,
    publish_release,
    release_folder,
)
from afabench.release.snapshot import (
    restore_snapshot,
    restored_manifest_location,
    save_snapshot,
)
from afabench.release.workflow_config import resolve_workflow_config

app = typer.Typer()
DEFAULT_OUTPUT_ROOT: Final[Path] = Path("extra/output")
DEFAULT_CHECKOUT: Final[Path] = Path()

type TransportFactory = Callable[[str], ReleaseTransport]

RepoIdOption = Annotated[
    str,
    typer.Option(
        envvar="AFABENCH_RELEASE_REPO",
        help="Hugging Face dataset repository holding the releases.",
    ),
]
TestReleaseOption = Annotated[
    bool,
    typer.Option(
        help="A test_only package, kept apart from official releases."
    ),
]


@app.command()
def save(
    snapshot_dir: Path,
    source_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    overwrite: bool = False,
    release_id: Annotated[
        str | None,
        typer.Option(help="Write a release manifest with this identity."),
    ] = None,
    scope: Annotated[
        ReleaseScope | None,
        typer.Option(help="Declared coverage; smoke outputs are test_only."),
    ] = None,
    profile: Annotated[
        Path | None,
        typer.Option(help="Snakemake profile the run used."),
    ] = None,
    configfile: Annotated[
        list[Path] | None,
        typer.Option(help="Workflow config file; replaces the profile's."),
    ] = None,
    config: Annotated[
        list[str] | None,
        typer.Option(help="KEY=VALUE override; replaces the profile's."),
    ] = None,
    checkout: Annotated[
        Path,
        typer.Option(help="Git checkout whose commit produced the outputs."),
    ] = DEFAULT_CHECKOUT,
) -> None:
    manifest = None
    if release_id is None:
        given = [
            name
            for name, value in [
                ("--scope", scope),
                ("--profile", profile),
                ("--configfile", configfile),
                ("--config", config),
            ]
            if value
        ]
        if given:
            msg = f"{', '.join(given)} only apply with --release-id."
            raise typer.BadParameter(msg)
    else:
        if scope is None:
            msg = "--release-id needs --scope (full, partial or test_only)."
            raise typer.BadParameter(msg)
        workflow_config = resolve_workflow_config(
            profile=profile,
            configfiles=configfile or [],
            overrides=config or [],
        )
        manifest = build_release_manifest(
            release_id=release_id,
            scope=scope,
            workflow_config=workflow_config,
            output_root=source_root,
            checkout=checkout,
        )
    save_snapshot(
        source_root, snapshot_dir, overwrite=overwrite, manifest=manifest
    )
    if manifest is not None:
        _echo_manifest(manifest, snapshot_dir / RELEASE_MANIFEST_FILENAME)


@app.command()
def restore(
    snapshot_dir: Path,
    destination_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    overwrite: bool = False,
) -> None:
    manifest_path = snapshot_dir / RELEASE_MANIFEST_FILENAME
    # Read before restoring so an unreadable manifest restores nothing.
    manifest = (
        read_release_manifest(manifest_path)
        if manifest_path.is_file()
        else None
    )
    restore_snapshot(snapshot_dir, destination_root, overwrite=overwrite)
    if manifest is not None:
        _echo_manifest(manifest, restored_manifest_location(destination_root))


@app.command()
def publish(
    ctx: typer.Context,
    snapshot_dir: Path,
    *,
    repo_id: RepoIdOption,
    test_release: TestReleaseOption = False,
) -> None:
    transport = _transport(ctx, repo_id)
    manifest = publish_release(
        snapshot_dir, transport, test_release=test_release
    )
    folder = release_folder(manifest.release_id, test_release=test_release)
    _echo_manifest(manifest, snapshot_dir / RELEASE_MANIFEST_FILENAME)
    typer.echo(f"Published to {transport.folder_url(folder)}")


@app.command()
def download(
    ctx: typer.Context,
    release_id: str,
    destination_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    repo_id: RepoIdOption,
    overwrite: bool = False,
    test_release: TestReleaseOption = False,
) -> None:
    manifest = download_release(
        release_id,
        _transport(ctx, repo_id),
        destination_root,
        overwrite=overwrite,
        test_release=test_release,
    )
    _echo_manifest(manifest, restored_manifest_location(destination_root))


def _transport(ctx: typer.Context, repo_id: str) -> ReleaseTransport:
    # Tests pass a fake transport factory as the context object.
    factory: TransportFactory = ctx.obj or HuggingFaceTransport
    return factory(repo_id)


def _echo_manifest(manifest: ReleaseManifest, path: Path) -> None:
    workflow_config = manifest.workflow_config
    configfiles = ", ".join(
        record.path for record in workflow_config.configfiles
    )
    dirty = " (dirty)" if manifest.code.dirty else ""
    typer.echo(
        f"Release {manifest.release_id}: scope {manifest.scope}, "
        f"execution {manifest.execution_mode}, "
        f"commit {manifest.code.commit}{dirty}\n"
        f"Workflow profile: {workflow_config.profile}\n"
        f"Workflow config files: {configfiles}\n"
        f"Workflow config overrides: {workflow_config.overrides}\n"
        f"Release manifest: {path}"
    )


if __name__ == "__main__":
    app()
