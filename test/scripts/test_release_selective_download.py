"""
Choose a release and download part of it through a fake release host.

The catalogs are synthetic: production-mode packages whose payloads are a
few bytes, declared `full` or `partial` only to exercise release selection.
They are not benchmark results.
"""

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from click.testing import Result
from typer.testing import CliRunner

from afabench.release.manifest import (
    PayloadCategory,
    ReleaseScope,
    build_release_manifest,
    read_release_manifest,
)
from afabench.release.workflow_config import WorkflowConfigRecord
from scripts.release.snapshot import app
from test.scripts.fake_release_transport import FakeReleaseTransport

runner = CliRunner()

REPO_ID = "afabench-test/releases"
TAG = "initializer-cold"
PLOT = f"plot_results/eval_split-test/{TAG}/cube/eval_perf.pdf"
# The external classifier of dataset realization 1 of cube.
EXTERNAL_CLASSIFIER = (
    f"trained_classifiers/{TAG}/dataset-cube+realization_index-1.bundle"
)
# What the pretraining and training jobs write beside their bundle.
TIME_RECORDS = {
    PayloadCategory.PRETRAINED_MODEL_BUNDLE: "pretrain_time.txt",
    PayloadCategory.AFA_METHOD_BUNDLE: "train_time.txt",
}


def workflow_config(*, smoke_test: bool = False) -> dict[str, Any]:
    """Alpha has no pretraining stage; beta pretrains and has a classifier."""
    return {
        "pretrain_mapping": {"shared": {"pretrain_script_name": "shared"}},
        "method_options": {
            "alpha": {"train_script_name": "alpha", "eval_batch_size": 4},
            "beta": {
                "train_script_name": "beta",
                "eval_batch_size": 4,
                "pretrained_model_name": "shared",
                "classifier": {"script_name": "special"},
            },
        },
        "methods": ["alpha", "beta"],
        "datasets": ["cube", "diabetes"],
        "dataset_realization_indices": [0, 1],
        "unmaskers": {"default": "direct"},
        "eval_hard_budgets": {"default": [3]},
        "soft_budget_params": {
            "alpha": {"default": [[0.5, 0.5]]},
            "beta": {"default": []},
        },
        "classifier_names": {"default": "masked_mlp_classifier"},
        "use_wandb": False,
        "smoke_test": smoke_test,
    }


def raw_table() -> pd.DataFrame:
    """External predictions only, so the builtin variant is not covered."""
    return pd.DataFrame(
        {
            "episode_id": [0, 0],
            "step": [0, 1],
            "action_performed": [3, 0],
            "builtin_predicted_class": pd.array([None, None], dtype="Int64"),
            "external_predicted_class": [1, 0],
            "true_class": [0, 0],
            "accumulated_cost": [1.0, 1.0],
            "forced_stop": [False, True],
            "eval_seed": [0, 0],
            "eval_hard_budget": [3.0, 3.0],
        }
    )


def write_release_outputs(
    root: Path,
    release_id: str,
    config: dict[str, Any],
    *,
    omit: Iterable[str] = (),
) -> None:
    """
    Write every table and bundle `config` schedules, except those in `omit`.

    Each transformed table, bundle and the plot holds the release id, so a
    test can tell which release a restored file came from.
    """
    # Only the scheduled paths are read from this manifest; scope smoke is
    # the scope every config, smoke or not, accepts.
    scheduled = build_release_manifest(
        release_id=release_id,
        scope=ReleaseScope.SMOKE,
        workflow_config=WorkflowConfigRecord(
            profile=None, configfiles=[], overrides={}, merged=config
        ),
        output_root=root,
        checkout=root,
    )
    omitted = set(omit)
    for table in scheduled.evaluation_tables:
        if table.raw_path not in omitted:
            (root / table.raw_path).parent.mkdir(parents=True)
            raw_table().to_parquet(root / table.raw_path, index=False)
        if table.transformed_path not in omitted:
            (root / table.transformed_path).parent.mkdir(parents=True)
            pd.DataFrame({"release": [release_id]}).to_parquet(
                root / table.transformed_path, index=False
            )
    for bundle in scheduled.bundles:
        if bundle.path in omitted:
            continue
        (root / bundle.path / "data").mkdir(parents=True)
        (root / bundle.path / "data/weights.bin").write_text(release_id)
        (root / bundle.path / "manifest.json").write_text(
            json.dumps({"bundle_version": "1.0.0", "class_name": "Fake"})
        )
        if bundle.category in TIME_RECORDS:
            time_record = TIME_RECORDS[bundle.category]
            (root / bundle.path).parent.joinpath(time_record).write_text("1.0")
    (root / PLOT).parent.mkdir(parents=True)
    (root / PLOT).write_text(release_id)


def invoke(transport: FakeReleaseTransport, *args: str) -> Result:
    return runner.invoke(
        app,
        [*args, "--repo-id", REPO_ID],
        obj=lambda _repo_id: transport,
    )


def publish(
    tmp_path: Path,
    transport: FakeReleaseTransport,
    release_id: str,
    *,
    scope: str = "full",
    created_at: str = "2026-10-01T00:00:00+00:00",
    smoke_test: bool = False,
    omit: Iterable[str] = (),
    backdate: bool = False,
) -> Path:
    """
    Save and publish a release whose manifest has `created_at`.

    `backdate` gives every saved output a distinct old mtime first.
    """
    config = workflow_config(smoke_test=smoke_test)
    source_root = tmp_path / f"{release_id}-source"
    write_release_outputs(source_root, release_id, config, omit=omit)
    configfile = tmp_path / f"{release_id}.yaml"
    configfile.write_text(yaml.safe_dump(config))
    # Every dataset is reviewed as permitted, so official releases publish.
    checkout = tmp_path / f"{release_id}-checkout"
    review_file = checkout / "extra/conf/release/dataset_redistribution.yaml"
    review_file.parent.mkdir(parents=True)
    review = {
        "status": "permitted",
        "license": "MIT",
        "source": "generated by afabench",
        "reviewed_by": "maintainer",
        "notes": None,
    }
    review_file.write_text(
        yaml.safe_dump({"datasets": dict.fromkeys(config["datasets"], review)})
    )
    package_dir = tmp_path / f"{release_id}-package"
    saved = runner.invoke(
        app,
        [
            "save",
            str(package_dir),
            "--source-root",
            str(source_root),
            "--configfile",
            str(configfile),
            "--release-id",
            release_id,
            "--scope",
            scope,
            "--checkout",
            str(checkout),
        ],
    )
    assert saved.exit_code == 0, saved.output
    # Synthetic creation times order the catalog independently of how fast
    # the test saves its packages.
    manifest_path = package_dir / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["created_at"] = created_at
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if backdate:
        paths = sorted((package_dir / "output").rglob("*"))
        for offset, path in enumerate(paths):
            mtime = 1_700_000_000 + offset
            os.utime(path, (mtime, mtime))
    published = invoke(
        transport,
        "publish",
        str(package_dir),
        *(["--smoke-release"] if scope == "smoke" else []),
    )
    assert published.exit_code == 0, published.output
    return package_dir


def download(
    transport: FakeReleaseTransport, destination_root: Path, *args: str
) -> Result:
    return invoke(
        transport,
        "download",
        *args,
        "--destination-root",
        str(destination_root),
    )


def restored_release_id(destination_root: Path) -> str:
    manifest_path = destination_root.parent / "release_manifest.json"
    return read_release_manifest(manifest_path).release_id


def restored_files(destination_root: Path) -> set[str]:
    return {
        path.relative_to(destination_root).as_posix()
        for path in destination_root.rglob("*")
        if path.is_file()
    }


def restored_bundles(destination_root: Path) -> set[str]:
    return {
        path.split(".bundle/")[0] + ".bundle"
        for path in restored_files(destination_root)
        if ".bundle/" in path
    }


def test_default_download_takes_the_latest_full_release_not_a_newer_partial(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    for release_id, scope, created_at in [
        ("2026-10-full", "full", "2026-10-01T00:00:00+00:00"),
        ("2026-11-partial", "partial", "2026-11-01T00:00:00+00:00"),
        ("2026-09-full", "full", "2026-09-01T00:00:00+00:00"),
    ]:
        publish(
            tmp_path / release_id,
            transport,
            release_id,
            scope=scope,
            created_at=created_at,
        )
    destination_root = tmp_path / "fork/extra/output"

    result = download(transport, destination_root, "--all")

    assert result.exit_code == 0, result.output
    assert restored_release_id(destination_root) == "2026-10-full"
    assert (destination_root / PLOT).read_text() == "2026-10-full"
    assert "Latest full release: 2026-10-full" in result.output


def test_default_download_without_a_full_release_names_the_partial_ones(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    publish(tmp_path, transport, "2026-11-partial", scope="partial")
    destination_root = tmp_path / "fork/extra/output"

    result = download(transport, destination_root, "--all")

    assert result.exit_code != 0
    message = str(result.exception)
    assert "No full release is published" in message
    assert "2026-11-partial (partial)" in message
    assert not destination_root.exists()


def test_older_and_partial_releases_download_when_named(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    for release_id, scope, created_at in [
        ("2026-09-full", "full", "2026-09-01T00:00:00+00:00"),
        ("2026-10-full", "full", "2026-10-01T00:00:00+00:00"),
        ("2026-11-partial", "partial", "2026-11-01T00:00:00+00:00"),
    ]:
        publish(
            tmp_path / release_id,
            transport,
            release_id,
            scope=scope,
            created_at=created_at,
        )

    for release_id, scope in [
        ("2026-09-full", "full"),
        ("2026-11-partial", "partial"),
    ]:
        destination_root = tmp_path / release_id / "fork/extra/output"
        result = download(transport, destination_root, release_id, "--all")
        assert result.exit_code == 0, result.output
        assert restored_release_id(destination_root) == release_id
        assert (destination_root / PLOT).read_text() == release_id
        assert f"Release {release_id}: scope {scope}" in result.output
        assert "Latest full release" not in result.output


@pytest.mark.parametrize(
    "selection",
    [
        ["--all"],
        ["--payload-category", "transformed_evaluation_table"]
        + ["--output-category", "plot_results"],
    ],
)
def test_a_release_published_during_a_download_is_not_mixed_in(
    tmp_path: Path, selection: list[str]
) -> None:
    transport = FakeReleaseTransport()
    publish(
        tmp_path / "old",
        transport,
        "2026-10-full",
        created_at="2026-10-01T00:00:00+00:00",
    )
    newer = tmp_path / "newer"
    published_newer: list[str] = []

    def publish_newer_release(path: str) -> None:
        # Right after the catalog is listed, as the first file is fetched,
        # a newer full release appears.
        if not published_newer:
            published_newer.append(path)
            publish(
                newer,
                transport,
                "2026-11-full",
                created_at="2026-11-01T00:00:00+00:00",
            )

    transport.before_download = publish_newer_release
    destination_root = tmp_path / "fork/extra/output"

    result = download(transport, destination_root, *selection)

    assert result.exit_code == 0, result.output
    assert published_newer
    assert "2026-11-full" in transport.list_folders("releases")
    assert restored_release_id(destination_root) == "2026-10-full"
    assert (destination_root / PLOT).read_text() == "2026-10-full"
    transformed = [
        path
        for path in restored_files(destination_root)
        if path.startswith("eval_results_transformed/")
    ]
    assert transformed
    for path in transformed:
        table = pd.read_parquet(destination_root / path)
        assert table["release"].tolist() == ["2026-10-full"]


def alpha_table(
    index: int, *, hard_budget: str = "3", soft_budget: str = "null"
) -> str:
    return (
        f"eval_split-test/{TAG}/alpha/dataset-cube+realization_index-{index}/"
        f"NO_PRETRAIN/train_seed-{index}+train_hard_budget-{hard_budget}+"
        f"train_soft_budget_param-{soft_budget}/"
        f"eval_seed-{index}+eval_hard_budget-{hard_budget}+"
        f"eval_soft_budget_param-{soft_budget}/eval_data.parquet"
    )


def requested_paths(transport: FakeReleaseTransport) -> list[str]:
    requested: list[str] = []
    transport.before_download = requested.append
    return requested


def test_results_only_selection_fetches_no_bundle(tmp_path: Path) -> None:
    transport = FakeReleaseTransport()
    publish(tmp_path, transport, "2026-10-full")
    requested = requested_paths(transport)
    destination_root = tmp_path / "fork/extra/output"

    result = download(
        transport,
        destination_root,
        "--payload-category",
        "raw_evaluation_table",
        "--payload-category",
        "transformed_evaluation_table",
        "--output-category",
        "plot_results",
        "--method",
        "alpha",
        "--dataset",
        "cube",
    )

    assert result.exit_code == 0, result.output
    tables = {
        f"{folder}/{alpha_table(index, **budgets)}"
        for folder in ["eval_results", "eval_results_transformed"]
        for index in [0, 1]
        for budgets in [
            {},
            {"hard_budget": "null", "soft_budget": "0.5"},
        ]
    }
    assert restored_files(destination_root) == {*tables, PLOT}
    assert not [path for path in requested if ".bundle" in path]
    assert "Missing from release" not in result.output
    assert restored_release_id(destination_root) == "2026-10-full"


def test_shared_prerequisites_download_without_afa_method_bundles(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    publish(tmp_path, transport, "2026-10-full")
    requested = requested_paths(transport)
    destination_root = tmp_path / "fork/extra/output"

    result = download(
        transport,
        destination_root,
        "--payload-category",
        "dataset_bundle",
        "--payload-category",
        "classifier_bundle",
        "--method",
        "alpha",
        "--dataset",
        "cube",
        "--dataset-realization",
        "1",
    )

    assert result.exit_code == 0, result.output
    bundles = restored_bundles(destination_root)
    # Each dataset realization has its own external classifier, so nothing
    # of dataset realization 0 comes along.
    assert bundles == {
        "datasets/cube/1/train.bundle",
        "datasets/cube/1/val.bundle",
        "datasets/cube/1/test.bundle",
        EXTERNAL_CLASSIFIER,
    }
    assert not [path for path in requested if "trained_methods/" in path]
    weights = destination_root / EXTERNAL_CLASSIFIER / "data/weights.bin"
    assert weights.read_text() == "2026-10-full"


def test_method_bundles_follow_the_selected_budget_setting(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    publish(tmp_path, transport, "2026-10-full")
    destination_root = tmp_path / "fork/extra/output"

    result = download(
        transport,
        destination_root,
        "--payload-category",
        "afa_method_bundle",
        "--payload-category",
        "pretrained_model_bundle",
        "--dataset",
        "cube",
        "--dataset-realization",
        "0",
        "--budget-setting",
        "soft_budget",
    )

    assert result.exit_code == 0, result.output
    bundles = restored_bundles(destination_root)
    # Only alpha has soft-budget evaluations, and it pretrains nothing.
    assert bundles == {
        f"trained_methods/{TAG}/alpha/dataset-cube+realization_index-0/"
        "NO_PRETRAIN/"
        "train_seed-0+train_hard_budget-null+train_soft_budget_param-0.5/"
        "method.bundle"
    }


def test_model_bundles_come_with_the_time_records_of_their_jobs(
    tmp_path: Path,
) -> None:
    # The workflow's time aggregation reads them: without them it would
    # rerun the pretraining or training job, replacing the restored bundle.
    transport = FakeReleaseTransport()
    publish(tmp_path, transport, "2026-10-full")
    destination_root = tmp_path / "fork/extra/output"

    result = download(
        transport,
        destination_root,
        "--payload-category",
        "pretrained_model_bundle",
        "--payload-category",
        "afa_method_bundle",
        "--method",
        "beta",
        "--dataset",
        "cube",
        "--dataset-realization",
        "0",
    )

    assert result.exit_code == 0, result.output
    assert {
        path
        for path in restored_files(destination_root)
        if ".bundle/" not in path
    } == {
        f"pretrained_models/{TAG}/shared/dataset-cube+realization_index-0/"
        "pretrain_seed-0/pretrain_time.txt",
        f"trained_methods/{TAG}/beta/dataset-cube+realization_index-0/"
        "pretrain_seed-0/"
        "train_seed-0+train_hard_budget-3+train_soft_budget_param-null/"
        "train_time.txt",
    }
    assert "Downloaded 0 file(s) and 2 folder(s)." in result.output


def beta_table(index: int) -> str:
    return (
        f"eval_split-test/{TAG}/beta/dataset-cube+realization_index-{index}/"
        f"pretrain_seed-{index}/"
        f"train_seed-{index}+train_hard_budget-3+train_soft_budget_param-null/"
        f"eval_seed-{index}+eval_hard_budget-3+eval_soft_budget_param-null/"
        "eval_data.parquet"
    )


def test_missing_coverage_is_reported_and_not_taken_from_another_release(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    publish(
        tmp_path / "full",
        transport,
        "2026-10-full",
        created_at="2026-10-01T00:00:00+00:00",
    )
    missing_table = f"eval_results_transformed/{beta_table(1)}"
    publish(
        tmp_path / "partial",
        transport,
        "2026-11-partial",
        scope="partial",
        created_at="2026-11-01T00:00:00+00:00",
        omit=[missing_table],
    )
    destination_root = tmp_path / "fork/extra/output"

    result = download(
        transport,
        destination_root,
        "2026-11-partial",
        "--payload-category",
        "transformed_evaluation_table",
        "--method",
        "beta",
        "--dataset",
        "cube",
        "--dataset",
        "physionet",
    )

    assert result.exit_code == 0, result.output
    assert "Missing from release 2026-11-partial:" in result.output
    assert "dataset 'physionet': no evaluation of the release matches" in (
        result.output
    )
    assert (
        f"transformed_evaluation_table {missing_table}: not in the release"
        in result.output
    )
    assert restored_files(destination_root) == {
        f"eval_results_transformed/{beta_table(0)}"
    }
    assert restored_release_id(destination_root) == "2026-11-partial"


def test_selection_the_release_does_not_cover_downloads_nothing(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    publish(tmp_path, transport, "2026-10-full")
    destination_root = tmp_path / "fork/extra/output"

    result = download(
        transport,
        destination_root,
        "--payload-category",
        "raw_evaluation_table",
        "--output-category",
        "combined_time_results",
        "--initializer",
        "warm",
        "--classifier-variant",
        "builtin",
    )

    assert result.exit_code != 0
    message = str(result.exception)
    assert "Nothing selected is in release '2026-10-full'" in message
    assert (
        "initializer 'warm': no evaluation of the release matches" in message
    )
    assert "classifier variant 'builtin'" in message
    assert "output category combined_time_results: not in the release" in (
        message
    )
    assert not destination_root.parent.exists()


def test_selective_download_refuses_to_overwrite_an_existing_output(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    package_dir = publish(tmp_path, transport, "2026-10-full")
    destination_root = tmp_path / "fork/extra/output"
    existing = destination_root / "eval_results_transformed" / alpha_table(0)
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"my own results")
    selection = [
        "--payload-category",
        "transformed_evaluation_table",
        "--payload-category",
        "dataset_bundle",
        "--method",
        "alpha",
    ]

    refused = download(transport, destination_root, *selection)

    assert refused.exit_code != 0
    assert str(existing) in str(refused.exception)
    assert existing.read_bytes() == b"my own results"
    assert restored_files(destination_root) == {
        existing.relative_to(destination_root).as_posix()
    }
    assert not (destination_root.parent / "release_manifest.json").exists()

    replaced = download(transport, destination_root, *selection, "--overwrite")

    assert replaced.exit_code == 0, replaced.output
    assert (
        existing.read_bytes()
        == (
            package_dir / "output/eval_results_transformed" / alpha_table(0)
        ).read_bytes()
    )
    assert (destination_root / "datasets/cube/1/test.bundle").is_dir()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--method", "alpha"],
        ["--all", "--payload-category", "raw_evaluation_table"],
        ["--all", "--dataset", "cube"],
    ],
)
def test_download_needs_either_all_or_a_category_selection(
    tmp_path: Path, arguments: list[str]
) -> None:
    transport = FakeReleaseTransport()
    publish(tmp_path, transport, "2026-10-full")
    requested = requested_paths(transport)
    destination_root = tmp_path / "fork/extra/output"

    result = download(transport, destination_root, *arguments)

    assert result.exit_code != 0
    assert "--payload-category" in result.output
    assert requested == []
    assert not destination_root.exists()


def test_selected_payloads_of_a_smoke_release_keep_smoke_provenance(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    package_dir = publish(
        tmp_path,
        transport,
        "smoke-check",
        scope="smoke",
        smoke_test=True,
    )
    destination_root = tmp_path / "fork/extra/output"

    result = download(
        transport,
        destination_root,
        "smoke-check",
        "--smoke-release",
        "--payload-category",
        "raw_evaluation_table",
        "--dataset",
        "cube",
    )

    assert result.exit_code == 0, result.output
    restored = read_release_manifest(
        destination_root.parent / "release_manifest.json"
    )
    assert restored == read_release_manifest(
        package_dir / "release_manifest.json"
    )
    assert "scope smoke, execution smoke" in result.output
    assert all(
        path.startswith("eval_results/")
        for path in restored_files(destination_root)
    )


def test_latest_is_not_a_smoke_release_and_not_a_release_id(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    publish(tmp_path, transport, "smoke-check", scope="smoke", smoke_test=True)

    result = download(
        transport, tmp_path / "fork/extra/output", "--smoke-release", "--all"
    )
    published = publish_reserved_release_id(tmp_path, transport)

    assert result.exit_code != 0
    assert "name a smoke release by its release id" in str(result.exception)
    assert published.exit_code != 0
    assert "reserved" in str(published.exception)
    assert transport.list_folders("releases") == []


def publish_reserved_release_id(
    tmp_path: Path, transport: FakeReleaseTransport
) -> Result:
    config = workflow_config()
    source_root = tmp_path / "latest-source"
    write_release_outputs(source_root, "latest", config)
    configfile = tmp_path / "latest.yaml"
    configfile.write_text(yaml.safe_dump(config))
    package_dir = tmp_path / "latest-package"
    runner.invoke(
        app,
        ["save", str(package_dir), "--source-root", str(source_root)]
        + ["--configfile", str(configfile), "--release-id", "latest"]
        + ["--scope", "full"],
    )
    return invoke(transport, "publish", str(package_dir))


def test_selected_outputs_keep_their_published_mtimes(tmp_path: Path) -> None:
    # Snakemake judges restored outputs, bundle folders included, by mtime.
    transport = FakeReleaseTransport()
    package_dir = publish(tmp_path, transport, "2026-10-full", backdate=True)
    destination_root = tmp_path / "fork/extra/output"

    result = download(
        transport,
        destination_root,
        "--payload-category",
        "transformed_evaluation_table",
        "--payload-category",
        "dataset_bundle",
        "--method",
        "alpha",
        "--dataset-realization",
        "0",
    )

    assert result.exit_code == 0, result.output
    bundle = "datasets/cube/0/test.bundle"
    for path in [
        f"eval_results_transformed/{alpha_table(0)}",
        bundle,
        f"{bundle}/data",
        f"{bundle}/data/weights.bin",
    ]:
        restored = destination_root / path
        published = package_dir / "output" / path
        assert restored.stat().st_mtime == published.stat().st_mtime, path
