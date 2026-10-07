"""
Save and restore an output snapshot (see `afabench.release.snapshot`).

`save --release-id` also writes a release manifest from the checkout and the
workflow configuration given by `--profile`, `--configfile` and `--config`
(see `docs/release_manifest.md`). `inventory` reports, for the same
configuration, how many payloads of each category an output root holds and
their size, without copying anything.

`publish` uploads such a snapshot to the release host and `download`
retrieves one release from it, the latest full one unless a release id is
given, either whole (`--all`) or the payloads selected by category and
coverage (see `afabench.release.publishing`, `afabench.release.selection`
and `docs/release_publishing.md`). Nothing else here touches the host.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Final

import typer

from afabench.release.huggingface import HuggingFaceTransport
from afabench.release.manifest import (
    RELEASE_MANIFEST_FILENAME,
    BudgetSetting,
    ClassifierVariant,
    PayloadCategory,
    RedistributionStatus,
    ReleaseManifest,
    ReleaseScope,
    build_release_manifest,
    inventory_payloads,
    read_release_manifest,
)
from afabench.release.publishing import (
    LATEST_RELEASE,
    ReleaseTransport,
    download_release,
    download_selection,
    publish_release,
    release_folder,
    resolve_release,
)
from afabench.release.selection import ReleaseSelection, select_payloads
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


@app.command()
def publish(
    ctx: typer.Context,
    snapshot_dir: Path,
    *,
    repo_id: RepoIdOption,
    test_release: TestReleaseOption = False,
    allow_redistribution: Annotated[
        list[str] | None,
        typer.Option(
            help="Publish this unreviewed or restricted dataset key in an "
            "official release anyway; recorded in the host's commit message."
        ),
    ] = None,
) -> None:
    transport = _transport(ctx, repo_id)
    manifest = publish_release(
        snapshot_dir,
        transport,
        test_release=test_release,
        allow_redistribution=allow_redistribution or [],
    )
    folder = release_folder(manifest.release_id, test_release=test_release)
    _echo_manifest(manifest, snapshot_dir / RELEASE_MANIFEST_FILENAME)
    typer.echo(f"Published to {transport.folder_url(folder)}")


@app.command()
def download(
    ctx: typer.Context,
    release: Annotated[
        str,
        typer.Argument(
            help=f"Release id, or {LATEST_RELEASE!r} for the latest full "
            "release."
        ),
    ] = LATEST_RELEASE,
    destination_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    repo_id: RepoIdOption,
    everything: Annotated[
        bool,
        typer.Option("--all", help="Download every file of the release."),
    ] = False,
    payload_category: Annotated[
        list[PayloadCategory] | None,
        typer.Option(help="Download this payload category's selected files."),
    ] = None,
    output_category: Annotated[
        list[str] | None,
        typer.Option(
            help="Download this top-level output folder, such as "
            "plot_results, whole."
        ),
    ] = None,
    dataset: Annotated[
        list[str] | None, typer.Option(help="Only this dataset key.")
    ] = None,
    method: Annotated[
        list[str] | None, typer.Option(help="Only this method.")
    ] = None,
    dataset_instance: Annotated[
        list[int] | None, typer.Option(help="Only this dataset instance.")
    ] = None,
    eval_split: Annotated[
        list[str] | None, typer.Option(help="Only this evaluation split.")
    ] = None,
    initializer: Annotated[
        list[str] | None, typer.Option(help="Only this initializer.")
    ] = None,
    budget_setting: Annotated[
        list[BudgetSetting] | None,
        typer.Option(help="Only hard-budget or soft-budget evaluations."),
    ] = None,
    classifier_variant: Annotated[
        list[ClassifierVariant] | None,
        typer.Option(help="Only evaluations with these predictions."),
    ] = None,
    overwrite: bool = False,
    test_release: TestReleaseOption = False,
) -> None:
    """
    Download all of one release, or the selected payloads of it.

    Coverage options select evaluations; payload categories then name what
    of them to fetch: their tables, and the bundles they depend on.
    """
    selection = ReleaseSelection(
        payload_categories=payload_category or [],
        output_categories=output_category or [],
        datasets=dataset or [],
        methods=method or [],
        dataset_instance_indices=dataset_instance or [],
        eval_splits=eval_split or [],
        initializers=initializer or [],
        budget_settings=budget_setting or [],
        classifier_variants=classifier_variant or [],
    )
    if everything == (selection != ReleaseSelection()):
        msg = (
            "Choose what to download: --all for every file of the release, "
            "or --payload-category/--output-category with optional coverage "
            "options, not both."
        )
        raise typer.BadParameter(msg)
    if not everything and not (payload_category or output_category):
        msg = (
            "Coverage options need --payload-category or --output-category "
            "to say what to download."
        )
        raise typer.BadParameter(msg)
    transport = _transport(ctx, repo_id)
    resolved = resolve_release(release, transport, test_release=test_release)
    manifest = resolved.manifest
    if release == LATEST_RELEASE:
        typer.echo(f"Latest full release: {manifest.release_id}")
    if everything:
        download_release(
            resolved, transport, destination_root, overwrite=overwrite
        )
    else:
        payloads = select_payloads(manifest, selection)
        download_selection(
            resolved,
            payloads,
            transport,
            destination_root,
            overwrite=overwrite,
        )
        typer.echo(
            f"Downloaded {len(payloads.files)} file(s) and "
            f"{len(payloads.folders)} folder(s)."
        )
        if payloads.missing:
            typer.echo(
                f"Missing from release {manifest.release_id}:\n  "
                + "\n  ".join(payloads.missing)
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
