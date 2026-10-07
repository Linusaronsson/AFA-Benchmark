import os
from pathlib import Path

from typer.testing import CliRunner

from scripts.release.snapshot import app

runner = CliRunner()


def build_output_tree(root: Path) -> None:
    bundle = root / "datasets/cube/0/train.bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"bundle_version": 1}')
    (bundle / ".snakemake_timestamp").write_text("")
    backdated = 1_700_000_000
    for path in root.rglob("*"):
        if path.is_file():
            os.utime(path, (backdated, backdated))


def test_save_then_restore_commands_round_trip_a_tree(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    snapshot_dir = tmp_path / "snapshot"
    destination_root = tmp_path / "destination"

    save_result = runner.invoke(
        app,
        ["save", str(snapshot_dir), "--source-root", str(source_root)],
    )
    assert save_result.exit_code == 0, save_result.output

    restore_result = runner.invoke(
        app,
        [
            "restore",
            str(snapshot_dir),
            "--destination-root",
            str(destination_root),
        ],
    )
    assert restore_result.exit_code == 0, restore_result.output

    restored = destination_root / "datasets/cube/0/train.bundle"
    assert (restored / "manifest.json").read_text() == '{"bundle_version": 1}'
    assert (restored / ".snakemake_timestamp").exists()


def test_save_refuses_a_conflicting_target_without_overwrite_flag(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    snapshot_dir = tmp_path / "snapshot"
    conflicting = (
        snapshot_dir / "output/datasets/cube/0/train.bundle/manifest.json"
    )
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text("pre-existing")

    result = runner.invoke(
        app,
        ["save", str(snapshot_dir), "--source-root", str(source_root)],
    )

    assert result.exit_code != 0
    assert conflicting.read_text() == "pre-existing"


def test_save_with_overwrite_flag_replaces_conflicting_file(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    snapshot_dir = tmp_path / "snapshot"
    conflicting = (
        snapshot_dir / "output/datasets/cube/0/train.bundle/manifest.json"
    )
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text("pre-existing")

    result = runner.invoke(
        app,
        [
            "save",
            str(snapshot_dir),
            "--source-root",
            str(source_root),
            "--overwrite",
        ],
    )

    assert result.exit_code == 0, result.output
    assert conflicting.read_text() == '{"bundle_version": 1}'
