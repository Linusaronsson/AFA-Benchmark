import os
from pathlib import Path

import pytest

from afabench.release.snapshot import restore_snapshot, save_snapshot


def build_output_tree(root: Path) -> None:
    bundle = root / "datasets/cube/0/train.bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text('{"bundle_version": 1}')
    (bundle / ".snakemake_timestamp").write_text("")
    (root / "plots/eval_perf.svg").parent.mkdir(parents=True)
    (root / "plots/eval_perf.svg").write_text("<svg></svg>")
    # Backdate mtimes so the round trip can prove they are preserved rather
    # than coincidentally fresh.
    backdated = 1_700_000_000
    for path in root.rglob("*"):
        if path.is_file():
            os.utime(path, (backdated, backdated))


def test_round_trip_reproduces_every_file_byte_for_byte_with_mtimes(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    snapshot_dir = tmp_path / "snapshot"
    destination_root = tmp_path / "destination"

    save_snapshot(source_root, snapshot_dir)
    restore_snapshot(snapshot_dir, destination_root)

    source_files = sorted(
        p.relative_to(source_root)
        for p in source_root.rglob("*")
        if p.is_file()
    )
    destination_files = sorted(
        p.relative_to(destination_root)
        for p in destination_root.rglob("*")
        if p.is_file()
    )
    assert source_files == destination_files
    for relative in source_files:
        source_file = source_root / relative
        destination_file = destination_root / relative
        assert destination_file.read_bytes() == source_file.read_bytes()
        assert destination_file.stat().st_mtime == source_file.stat().st_mtime


def test_round_trip_preserves_directory_mtimes_including_empty_directories(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    empty_bundle = source_root / "datasets/cube/0/val.bundle"
    empty_bundle.mkdir(parents=True)
    backdated = 1_650_000_000
    os.utime(empty_bundle, (backdated, backdated))
    os.utime(
        source_root / "datasets/cube/0/train.bundle", (backdated, backdated)
    )
    os.utime(source_root / "datasets/cube/0", (backdated, backdated))
    snapshot_dir = tmp_path / "snapshot"
    destination_root = tmp_path / "destination"

    save_snapshot(source_root, snapshot_dir)
    restore_snapshot(snapshot_dir, destination_root)

    assert (destination_root / "datasets/cube/0/val.bundle").is_dir()
    assert (
        destination_root / "datasets/cube/0/val.bundle"
    ).stat().st_mtime == (empty_bundle.stat().st_mtime)
    assert (
        destination_root / "datasets/cube/0/train.bundle"
    ).stat().st_mtime == (
        source_root / "datasets/cube/0/train.bundle"
    ).stat().st_mtime
    # Creating the empty bundle must not bump its parent's restored mtime.
    assert (destination_root / "datasets/cube/0").stat().st_mtime == (
        source_root / "datasets/cube/0"
    ).stat().st_mtime


def test_save_refuses_a_conflicting_target_and_writes_nothing(
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
    unrelated = snapshot_dir / "output/unrelated.txt"
    unrelated.write_text("keep me")

    with pytest.raises(FileExistsError, match=str(conflicting)):
        save_snapshot(source_root, snapshot_dir)

    assert conflicting.read_text() == "pre-existing"
    assert unrelated.read_text() == "keep me"
    assert not (snapshot_dir / "output/plots").exists()


def test_save_with_overwrite_replaces_only_conflicting_files(
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
    unrelated = snapshot_dir / "output/unrelated.txt"
    unrelated.write_text("keep me")

    save_snapshot(source_root, snapshot_dir, overwrite=True)

    assert (
        conflicting.read_text()
        == (
            source_root / "datasets/cube/0/train.bundle/manifest.json"
        ).read_text()
    )
    assert unrelated.read_text() == "keep me"
    assert (snapshot_dir / "output/plots/eval_perf.svg").exists()


def test_restore_refuses_a_conflicting_target_and_writes_nothing(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    snapshot_dir = tmp_path / "snapshot"
    save_snapshot(source_root, snapshot_dir)
    destination_root = tmp_path / "destination"
    conflicting = (
        destination_root / "datasets/cube/0/train.bundle/manifest.json"
    )
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text("pre-existing")
    unrelated = destination_root / "unrelated.txt"
    unrelated.write_text("keep me")

    with pytest.raises(FileExistsError, match=str(conflicting)):
        restore_snapshot(snapshot_dir, destination_root)

    assert conflicting.read_text() == "pre-existing"
    assert unrelated.read_text() == "keep me"
    assert not (destination_root / "plots").exists()


def test_restore_with_overwrite_replaces_only_conflicting_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    build_output_tree(source_root)
    snapshot_dir = tmp_path / "snapshot"
    save_snapshot(source_root, snapshot_dir)
    destination_root = tmp_path / "destination"
    conflicting = (
        destination_root / "datasets/cube/0/train.bundle/manifest.json"
    )
    conflicting.parent.mkdir(parents=True)
    conflicting.write_text("pre-existing")
    unrelated = destination_root / "unrelated.txt"
    unrelated.write_text("keep me")

    restore_snapshot(snapshot_dir, destination_root, overwrite=True)

    assert (
        conflicting.read_text()
        == (
            source_root / "datasets/cube/0/train.bundle/manifest.json"
        ).read_text()
    )
    assert unrelated.read_text() == "keep me"
    assert (destination_root / "plots/eval_perf.svg").exists()


def test_missing_source_root_is_an_error_naming_the_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError, match=str(missing)):
        save_snapshot(missing, tmp_path / "snapshot")


def test_empty_source_root_is_an_error_naming_the_path(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match=str(empty)):
        save_snapshot(empty, tmp_path / "snapshot")

    assert not (tmp_path / "snapshot").exists()
