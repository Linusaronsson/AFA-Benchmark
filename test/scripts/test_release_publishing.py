"""Publish and download benchmark releases through a fake release host."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from click.testing import Result
from typer.testing import CliRunner

from afabench.release.manifest import (
    ExecutionMode,
    ReleaseScope,
    read_release_manifest,
)
from scripts.release.snapshot import app
from test.scripts.fake_release_transport import FakeReleaseTransport

runner = CliRunner()

REPO_ID = "afabench-test/releases"
TABLE = (
    "eval_split-test/initializer-cold/alpha/"
    "dataset-cube+instance_idx-0/NO_PRETRAIN/"
    "train_seed-0+train_hard_budget-3+train_soft_budget_param-null/"
    "eval_seed-0+eval_hard_budget-3+eval_soft_budget_param-null/"
    "eval_data.parquet"
)
RAW_TABLE = f"eval_results/{TABLE}"
TRANSFORMED_TABLE = f"eval_results_transformed/{TABLE}"
PLOT = "plot_results/eval_split-test/initializer-cold/cube/eval_perf.pdf"


def workflow_config(*, smoke_test: bool) -> dict[str, Any]:
    return {
        "pretrain_mapping": {},
        "method_options": {
            "alpha": {"train_script_name": "alpha", "eval_batch_size": 4}
        },
        "methods": ["alpha"],
        "datasets": ["cube"],
        "dataset_instance_indices": [0],
        "unmaskers": {"default": "direct"},
        "eval_hard_budgets": {"default": [3]},
        "soft_budget_params": {"alpha": {"default": []}},
        "classifier_names": {"default": "masked_mlp_classifier"},
        "use_wandb": False,
        "smoke_test": smoke_test,
    }


def raw_table() -> pd.DataFrame:
    """One forced-stop episode with external predictions only."""
    return pd.DataFrame(
        {
            "episode_id": pd.array([0, 0, 0], dtype="Int64"),
            "step": pd.array([0, 1, 2], dtype="Int64"),
            "action_performed": pd.array([3, 1, 0], dtype="Int64"),
            "builtin_predicted_class": pd.array(
                [None, None, None], dtype="Int64"
            ),
            "external_predicted_class": pd.array([1, 0, 0], dtype="Int64"),
            "true_class": pd.array([0, 0, 0], dtype="Int64"),
            "accumulated_cost": [1.0, 2.0, 2.0],
            "forced_stop": [False, False, True],
            "eval_seed": pd.array([0, 0, 0], dtype="Int64"),
            "eval_hard_budget": [3.0, 3.0, 3.0],
        }
    )


def transformed_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "afa_method": ["alpha", "alpha"],
            "classifier": ["builtin", "external"],
            "predicted_class": pd.array([None, 1], dtype="Int64"),
            "true_class": pd.array([0, 0], dtype="Int64"),
            "n_selections_performed": pd.array([1, 1], dtype="Int64"),
            "eval_soft_budget_param": [None, None],
        }
    )


def build_output_tree(root: Path, *, plot: bytes = b"%PDF-1.4 plot") -> None:
    bundle = root / "datasets/cube/0/train.bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"bundle_version": 1}')
    for path, table in [
        (RAW_TABLE, raw_table()),
        (TRANSFORMED_TABLE, transformed_table()),
    ]:
        (root / path).parent.mkdir(parents=True)
        table.to_parquet(root / path, index=False)
    (root / PLOT).parent.mkdir(parents=True)
    (root / PLOT).write_bytes(plot)


def prepare_package(
    tmp_path: Path,
    release_id: str,
    *,
    scope: str = "full",
    smoke_test: bool = False,
    plot: bytes = b"%PDF-1.4 plot",
) -> Path:
    """Save a snapshot with a release manifest, as a maintainer would."""
    source_root = tmp_path / f"{release_id}-source"
    build_output_tree(source_root, plot=plot)
    configfile = tmp_path / f"{release_id}.yaml"
    configfile.write_text(
        yaml.safe_dump(workflow_config(smoke_test=smoke_test))
    )
    package_dir = tmp_path / f"{release_id}-package"
    result = runner.invoke(
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
        ],
    )
    assert result.exit_code == 0, result.output
    return package_dir


def invoke(transport: FakeReleaseTransport, *args: str) -> Result:
    return runner.invoke(
        app,
        [*args, "--repo-id", REPO_ID],
        obj=lambda _repo_id: transport,
    )


def download(
    transport: FakeReleaseTransport,
    release_id: str,
    destination_root: Path,
    *extra: str,
) -> Result:
    return invoke(
        transport,
        "download",
        release_id,
        "--all",
        "--destination-root",
        str(destination_root),
        *extra,
    )


def test_published_release_downloads_with_native_tables_and_plots(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    package_dir = prepare_package(tmp_path, "2026-10-cube")
    destination_root = tmp_path / "fork/extra/output"

    published = invoke(transport, "publish", str(package_dir))
    downloaded = download(transport, "2026-10-cube", destination_root)

    assert published.exit_code == 0, published.output
    assert downloaded.exit_code == 0, downloaded.output
    source_root = package_dir / "output"
    for path in [RAW_TABLE, TRANSFORMED_TABLE, PLOT]:
        assert (destination_root / path).read_bytes() == (
            source_root / path
        ).read_bytes()
    pd.testing.assert_frame_equal(
        pd.read_parquet(destination_root / RAW_TABLE), raw_table()
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(destination_root / TRANSFORMED_TABLE),
        transformed_table(),
    )
    manifest = read_release_manifest(
        destination_root.parent / "release_manifest.json"
    )
    assert manifest.release_id == "2026-10-cube"
    assert manifest.evaluation_tables[0].raw_present
    assert "Release 2026-10-cube" in downloaded.output


def test_test_only_package_is_refused_as_an_official_release(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    package_dir = prepare_package(
        tmp_path, "smoke-check", scope="test_only", smoke_test=True
    )

    result = invoke(transport, "publish", str(package_dir))

    assert result.exit_code != 0
    assert "test_only" in str(result.exception)
    assert transport.files == {}


def test_smoke_provenance_survives_a_test_release_round_trip(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    package_dir = prepare_package(
        tmp_path, "smoke-check", scope="test_only", smoke_test=True
    )
    destination_root = tmp_path / "fork/extra/output"

    published = invoke(
        transport, "publish", str(package_dir), "--test-release"
    )
    downloaded = download(
        transport, "smoke-check", destination_root, "--test-release"
    )

    assert published.exit_code == 0, published.output
    assert downloaded.exit_code == 0, downloaded.output
    restored = read_release_manifest(
        destination_root.parent / "release_manifest.json"
    )
    assert restored == read_release_manifest(
        package_dir / "release_manifest.json"
    )
    assert restored.scope is ReleaseScope.TEST_ONLY
    assert restored.execution_mode is ExecutionMode.SMOKE
    assert "scope test_only, execution smoke" in downloaded.output


def test_test_release_is_not_downloadable_as_an_official_release(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    package_dir = prepare_package(
        tmp_path, "smoke-check", scope="test_only", smoke_test=True
    )
    invoke(transport, "publish", str(package_dir), "--test-release")
    destination_root = tmp_path / "fork/extra/output"

    result = download(transport, "smoke-check", destination_root)

    assert result.exit_code != 0
    assert "smoke-check" in str(result.exception)
    assert not destination_root.exists()


def test_test_release_flag_does_not_publish_an_official_package(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    package_dir = prepare_package(tmp_path, "2026-10-cube", scope="full")

    result = invoke(transport, "publish", str(package_dir), "--test-release")

    assert result.exit_code != 0
    assert transport.files == {}


def test_each_published_release_downloads_its_own_outputs(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    for release_id, plot in [("2026-10", b"October"), ("2026-11", b"Nov")]:
        package_dir = prepare_package(tmp_path, release_id, plot=plot)
        result = invoke(transport, "publish", str(package_dir))
        assert result.exit_code == 0, result.output

    for release_id, plot in [("2026-10", b"October"), ("2026-11", b"Nov")]:
        destination_root = tmp_path / release_id / "extra/output"
        result = download(transport, release_id, destination_root)
        assert result.exit_code == 0, result.output
        assert (destination_root / PLOT).read_bytes() == plot
        manifest = read_release_manifest(
            destination_root.parent / "release_manifest.json"
        )
        assert manifest.release_id == release_id


def test_a_published_release_is_not_replaced_by_publishing_again(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    first = prepare_package(tmp_path / "first", "2026-10", plot=b"first")
    second = prepare_package(tmp_path / "second", "2026-10", plot=b"second")
    invoke(transport, "publish", str(first))

    result = invoke(transport, "publish", str(second))

    assert result.exit_code != 0
    assert "2026-10" in str(result.exception)
    destination_root = tmp_path / "fork/extra/output"
    download(transport, "2026-10", destination_root)
    assert (destination_root / PLOT).read_bytes() == b"first"


def test_download_refuses_to_overwrite_an_existing_local_output(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    invoke(transport, "publish", str(prepare_package(tmp_path, "2026-10")))
    destination_root = tmp_path / "fork/extra/output"
    existing = destination_root / TRANSFORMED_TABLE
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"my own results")

    result = download(transport, "2026-10", destination_root)

    assert result.exit_code != 0
    assert str(existing) in str(result.exception)
    assert existing.read_bytes() == b"my own results"
    assert not (destination_root / RAW_TABLE).exists()
    assert not (destination_root.parent / "release_manifest.json").exists()


def test_download_with_overwrite_replaces_an_existing_local_output(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    package_dir = prepare_package(tmp_path, "2026-10")
    invoke(transport, "publish", str(package_dir))
    destination_root = tmp_path / "fork/extra/output"
    existing = destination_root / TRANSFORMED_TABLE
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"my own results")

    result = download(transport, "2026-10", destination_root, "--overwrite")

    assert result.exit_code == 0, result.output
    assert (
        existing.read_bytes()
        == (package_dir / "output" / TRANSFORMED_TABLE).read_bytes()
    )


def test_downloaded_outputs_keep_the_snapshot_directories_and_mtimes(
    tmp_path: Path,
) -> None:
    # Snakemake judges restored outputs by mtime, so a download must leave
    # the tree as the snapshot recorded it, although the host keeps bytes
    # only.
    transport = FakeReleaseTransport()
    package_dir = prepare_package(tmp_path, "2026-10")
    source_root = package_dir / "output"
    (source_root / "plot_results/empty").mkdir()
    paths = sorted(source_root.rglob("*"))
    for offset, path in enumerate(paths):
        backdated = 1_700_000_000 + offset
        os.utime(path, (backdated, backdated))
    invoke(transport, "publish", str(package_dir))
    destination_root = tmp_path / "fork/extra/output"

    result = download(transport, "2026-10", destination_root)

    assert result.exit_code == 0, result.output
    for path in paths:
        restored = destination_root / path.relative_to(source_root)
        assert restored.stat().st_mtime == path.stat().st_mtime, restored


def test_published_tables_and_plots_are_ordinary_files_on_the_host(
    tmp_path: Path,
) -> None:
    # Results-only researchers fetch single files without AFABench, so each
    # table and plot must be stored as-is, not inside an archive.
    transport = FakeReleaseTransport()
    package_dir = prepare_package(tmp_path, "2026-10")

    result = invoke(transport, "publish", str(package_dir))

    assert result.exit_code == 0, result.output
    for path in [RAW_TABLE, TRANSFORMED_TABLE, PLOT]:
        content = (package_dir / "output" / path).read_bytes()
        assert any(
            stored_path.endswith(f"/{path}") and stored == content
            for stored_path, stored in transport.files.items()
        ), path
    assert "https://hub.invalid/" in result.output


def test_snapshot_without_release_manifest_is_not_published(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    package_dir = tmp_path / "package"
    runner.invoke(
        app,
        ["save", str(package_dir), "--source-root", str(source_root)],
    )

    result = invoke(transport, "publish", str(package_dir))

    assert result.exit_code != 0
    assert "release_manifest.json" in str(result.exception)
    assert transport.files == {}


def test_release_id_that_is_not_a_single_path_segment_is_refused(
    tmp_path: Path,
) -> None:
    transport = FakeReleaseTransport()
    package_dir = prepare_package(tmp_path, "nested/2026-10")

    result = invoke(transport, "publish", str(package_dir))

    assert result.exit_code != 0
    assert "nested/2026-10" in str(result.exception)
    assert transport.files == {}


def test_saving_a_release_snapshot_uploads_nothing(tmp_path: Path) -> None:
    transport = FakeReleaseTransport()
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    configfile = tmp_path / "run.yaml"
    configfile.write_text(yaml.safe_dump(workflow_config(smoke_test=False)))

    result = runner.invoke(
        app,
        [
            "save",
            str(tmp_path / "package"),
            "--source-root",
            str(source_root),
            "--configfile",
            str(configfile),
            "--release-id",
            "2026-10",
            "--scope",
            "full",
        ],
        obj=lambda _repo_id: transport,
    )

    assert result.exit_code == 0, result.output
    assert transport.files == {}
