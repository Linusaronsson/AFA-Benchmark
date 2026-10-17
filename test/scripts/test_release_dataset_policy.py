"""Dataset onboarding policy and hash-proven exclusions from releases."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from afabench.release.manifest import (
    DATASET_REDISTRIBUTION_FILE,
    ArtifactIndex,
    InputRole,
    PayloadCategory,
    ReleaseManifest,
    build_release_manifest,
    dangling_inputs,
    dataset_redistribution,
    index_artifacts,
    read_release_manifest,
)
from afabench.release.publishing import publish_release
from afabench.release.selection import (
    CHAIN_CATEGORIES,
    ReleaseSelection,
    select_payloads,
)
from afabench.release.snapshot import save_snapshot
from afabench.release.workflow_config import WorkflowConfigRecord
from scripts.release.snapshot import app
from test.scripts.fake_release_transport import FakeReleaseTransport
from test.scripts.release_artifacts import (
    ALPHA,
    Catalog,
    dataset_bundle,
    provenance,
    record_run_config,
    write_bundle,
    write_catalog,
)

runner = CliRunner()


@pytest.fixture
def release_source(tmp_path: Path) -> tuple[Path, Path, ArtifactIndex]:
    source = tmp_path / "source"
    write_catalog(
        source,
        Catalog(
            methods=[ALPHA], datasets=["cube", "diabetes", "mnist", "actg"]
        ),
    )
    # An excluded bundle without any consumers must also be remembered.
    write_bundle(
        source,
        dataset_bundle("cube", 7, "train"),
        provenance(
            "dataset_generation",
            dataset_key="cube",
            dataset_realization_index=7,
            split="train",
        ),
    )
    checkout = tmp_path / "checkout"
    policy_file = checkout / DATASET_REDISTRIBUTION_FILE
    policy_file.parent.mkdir(parents=True)
    policy_file.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "cube": {
                        "status": "permitted",
                        "bundle_policy": "omit",
                        "omission_reason": "synthetic dataset; regenerate locally",
                    },
                    "diabetes": {
                        "status": "restricted",
                        "bundle_policy": "include",
                    },
                    "mnist": {
                        "status": "permitted",
                        "bundle_policy": "include",
                    },
                    "actg": {
                        "status": "unreviewed",
                        "bundle_policy": "include",
                    },
                }
            }
        )
    )
    return source, checkout, index_artifacts(source)


@pytest.fixture
def manifest(
    release_source: tuple[Path, Path, ArtifactIndex],
) -> ReleaseManifest:
    source, checkout, index = release_source
    return build_release_manifest(
        release_id="policy-test",
        workflow_config=WorkflowConfigRecord(
            profile=None, configfiles=[], overrides={}, merged={}
        ),
        output_root=source,
        index=index,
        checkout=checkout,
    )


def test_every_pipeline_dataset_has_an_explicit_valid_policy() -> None:
    checkout = Path(__file__).parents[2]
    datasets = {
        p.stem
        for p in (checkout / "conf/components/dataset_key").glob("*.yaml")
    }
    datasets.update(
        p.stem
        for p in (checkout / "conf/scripts/dataset_generation/dataset").glob(
            "*.yaml"
        )
    )
    for path in (checkout / "workflow/conf/datasets").glob("*.yaml"):
        datasets.update(yaml.safe_load(path.read_text())["datasets"])
    records = dataset_redistribution(checkout, sorted(datasets))
    assert set(records) == datasets


def test_policy_exclusions_preserve_exact_bundles_and_reasons(
    manifest: ReleaseManifest,
    release_source: tuple[Path, Path, ArtifactIndex],
) -> None:
    _, _, index = release_source
    omitted = manifest.omitted_dataset_bundles
    assert len(omitted) == 10
    assert [o.bundle for o in omitted] == [
        b
        for b in index.bundles
        if b.category is PayloadCategory.DATASET_BUNDLE
        and b.dataset_key != "mnist"
    ]
    assert {o.reason for o in omitted} == {
        "synthetic dataset; regenerate locally",
        "restricted redistribution permission",
        "unreviewed redistribution permission",
    }
    assert {
        b.dataset_key
        for b in manifest.bundles
        if b.category is PayloadCategory.DATASET_BUNDLE
    } == {"mnist"}
    assert dangling_inputs(manifest) == []
    selected = select_payloads(
        manifest,
        ReleaseSelection(
            payload_categories=CHAIN_CATEGORIES, datasets=["cube"]
        ),
    )
    assert selected.missing == []
    assert len(selected.omitted_dataset_bundles) == 3
    assert {
        o.bundle.dataset_key for o in selected.omitted_dataset_bundles
    } == {"cube"}
    assert not any("datasets/cube" in path for path in selected.folders)
    results_only = select_payloads(
        manifest,
        ReleaseSelection(
            payload_categories=[PayloadCategory.RAW_EVALUATION_TABLE]
        ),
    )
    assert results_only.omitted_dataset_bundles == []


@pytest.mark.parametrize(
    "fault",
    [
        "different_hash",
        "null_hash",
        "wrong_role",
        "absent_inventory",
        "null_inventory_hash",
    ],
)
def test_only_proven_dataset_exclusions_are_excused(
    manifest: ReleaseManifest, fault: str
) -> None:
    consumer = next(b for b in manifest.bundles if b.dataset_key == "cube")
    original = consumer.inputs[0]
    entry = replace(
        original,
        content_hash={
            "different_hash": "sha256:changed",
            "null_hash": None,
        }.get(fault, original.content_hash),
        role=InputRole.CLASSIFIER if fault == "wrong_role" else original.role,
    )
    consumer = replace(consumer, inputs=[entry])
    omitted = manifest.omitted_dataset_bundles
    if fault == "absent_inventory":
        omitted = []
    elif fault == "null_inventory_hash":
        omitted = [
            replace(o, bundle=replace(o.bundle, content_hash=None))
            for o in omitted
        ]
    manifest = replace(
        manifest,
        bundles=[consumer],
        evaluations=[replace(manifest.evaluations[0], inputs=[entry])],
        omitted_dataset_bundles=omitted,
    )
    dangling = dangling_inputs(manifest)
    assert len(dangling) == 2
    assert {value.path for value in dangling} == {original.path}
    selected = select_payloads(
        manifest, ReleaseSelection(payload_categories=CHAIN_CATEGORIES)
    )
    assert len(selected.missing) == 1
    assert original.path in selected.missing[0]
    assert selected.omitted_dataset_bundles == []


def test_cli_groups_exclusions_and_round_trips_them(
    tmp_path: Path, release_source: tuple[Path, Path, ArtifactIndex]
) -> None:
    source, checkout, _ = release_source
    configfile = tmp_path / "run.yaml"
    configfile.write_text("smoke_test: false\n")
    record_run_config(source, configfile)
    package = tmp_path / "package"
    saved = runner.invoke(
        app,
        [
            "save",
            str(package),
            "--source-root",
            str(source),
            "--release-id",
            "policy-test",
            "--checkout",
            str(checkout),
        ],
    )
    assert saved.exit_code == 0, saved.exception
    assert "10 excluded by policy" in saved.output
    assert saved.output.count("cube: 4 bundle(s) — synthetic") == 1
    assert (
        "diabetes: 3 bundle(s) — restricted redistribution permission"
        in saved.output
    )
    assert (
        "actg: 3 bundle(s) — unreviewed redistribution permission"
        in saved.output
    )
    assert "Inputs no bundle" not in saved.output
    assert (
        "Omitted dataset bundles without redistribution permission"
        not in saved.output
    )
    manifest = read_release_manifest(package / "release_manifest.json")
    assert len(manifest.omitted_dataset_bundles) == 10
    transport = FakeReleaseTransport()
    published = runner.invoke(
        app,
        ["publish", str(package), "--repo-id", "fake/repo"],
        obj=lambda _: transport,
    )
    assert published.exit_code == 0, published.exception
    assert "cube: 4 bundle(s) — synthetic" in published.output
    for options, count in [(["--all"], 10), (["--dataset", "cube"], 3)]:
        destination = tmp_path / f"download-{count}"
        downloaded = runner.invoke(
            app,
            [
                "download",
                "policy-test",
                "--repo-id",
                "fake/repo",
                "--destination-root",
                str(destination),
                *options,
            ],
            obj=lambda _: transport,
        )
        assert downloaded.exit_code == 0, downloaded.exception
        assert f"{count} excluded by policy" in downloaded.output
        assert "Missing from release" not in downloaded.output
        assert (
            read_release_manifest(destination / "release_manifest.json")
            == manifest
        )
    restored = runner.invoke(
        app,
        [
            "restore",
            str(package),
            "--destination-root",
            str(tmp_path / "restored"),
        ],
    )
    assert restored.exit_code == 0, restored.exception
    assert "10 excluded by policy" in restored.output


def test_publication_refuses_permitted_but_policy_excluded_bundles(
    tmp_path: Path,
    release_source: tuple[Path, Path, ArtifactIndex],
    manifest: ReleaseManifest,
) -> None:
    source, _, _ = release_source
    package = tmp_path / "package"
    save_snapshot(source, package, manifest=manifest)
    bundle_path = dataset_bundle("cube", 0, "train")
    shutil.copytree(source / bundle_path, package / "output" / bundle_path)
    transport = FakeReleaseTransport()
    with pytest.raises(ValueError, match="added: output/datasets/cube"):
        publish_release(package, transport)
    assert transport.files == {}


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_explicit_omission_requires_a_reason(
    tmp_path: Path, reason: str | None
) -> None:
    policy_file = tmp_path / DATASET_REDISTRIBUTION_FILE
    policy_file.parent.mkdir(parents=True)
    policy_file.write_text(
        yaml.safe_dump(
            {
                "datasets": {
                    "cube": {
                        "status": "permitted",
                        "bundle_policy": "omit",
                        "omission_reason": reason,
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="requires omission_reason"):
        dataset_redistribution(tmp_path, ["cube"])
