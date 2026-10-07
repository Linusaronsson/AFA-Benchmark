"""
Save and restore an output snapshot (see `afabench.release.snapshot`).

`save --release-id` also writes a release manifest from the checkout and the
workflow configuration given by `--profile`, `--configfile` and `--config`
(see `docs/release_manifest.md`). `inventory` reports, for the same
configuration, how many payloads of each category an output root holds and
their size, without copying anything.
"""

from pathlib import Path
from typing import Annotated, Final

import typer

from afabench.release.manifest import (
    RELEASE_MANIFEST_FILENAME,
    RedistributionStatus,
    ReleaseManifest,
    ReleaseScope,
    build_release_manifest,
    inventory_payloads,
    read_release_manifest,
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
def inventory(
    source_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
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
) -> None:
    """Report each payload category's count and size; copies nothing."""
    payload_inventory = inventory_payloads(
        workflow_config=resolve_workflow_config(
            profile=profile,
            configfiles=configfile or [],
            overrides=config or [],
        ),
        output_root=source_root,
    )
    typer.echo(f"Execution: {payload_inventory.execution_mode}")
    for payload in payload_inventory.payloads:
        typer.echo(
            f"{payload.category}: {payload.present}/{payload.scheduled} "
            f"present, {payload.size_bytes} bytes, "
            f"classes: {', '.join(payload.class_names) or 'none'}"
        )


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
    for status in [
        RedistributionStatus.UNREVIEWED,
        RedistributionStatus.RESTRICTED,
    ]:
        datasets = [
            dataset
            for dataset, review in (
                manifest.settings.dataset_redistribution.items()
            )
            if review.status is status
        ]
        if datasets:
            typer.echo(
                f"{status.capitalize()} dataset redistribution: "
                f"{', '.join(datasets)}"
            )


if __name__ == "__main__":
    app()
