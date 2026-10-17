"""Release packages carry a browsable summary of their actual contents."""

import hashlib
from pathlib import Path

import pytest

from afabench.release.manifest import read_release_manifest
from afabench.release.publishing import release_files
from afabench.release.snapshot import restore_snapshot
from scripts.release.snapshot import app
from test.scripts.release_artifacts import ALPHA, Catalog, write_catalog
from test.scripts.test_release_manifest import runner, save, save_catalog


def test_release_readme_describes_tables_and_is_inventoried(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    write_catalog(source, Catalog(methods=[ALPHA]))
    plots = source / "plots"
    plots.mkdir()
    (plots / "accuracy.svg").write_text("<svg/>")
    notes = tmp_path / "notes.md"
    notes.write_text("Initial results.\n\n**Caveat:** limited coverage.")

    result = save(tmp_path, "--release-notes", str(notes))

    assert result.exit_code == 0, result.output
    package = tmp_path / "snapshot"
    readme = package / "README.md"
    text = readme.read_text()
    manifest = read_release_manifest(package / "release_manifest.json")
    assert "# AFABench release 2026-10-cube" in text
    assert "**Caveat:** limited coverage." in text
    assert "- Datasets: cube" in text
    assert "- Methods: alpha" in text
    assert f"- Evaluations: {len(manifest.evaluations)}" in text
    assert f"| cube | alpha | {len(manifest.evaluations)} |" in text
    assert "- Evaluation splits: test" in text
    assert "- Training seeds: 0" in text
    assert "- Evaluation hard budgets: 3.0" in text
    assert "- Plot files: 1" in text
    assert "[plots](./output/plots/)" in text
    assert "[full release manifest](./release_manifest.json)" in text
    assert "not every combination" in text
    for entry in manifest.evaluations:
        assert entry.raw_path is not None
    assert "[example table](./output/eval_results/" in text
    assert "[example table](./output/eval_results_transformed/" in text
    assert (
        manifest.file_inventory["README.md"]
        == hashlib.sha256(readme.read_bytes()).hexdigest()
    )
    assert "releases/2026-10-cube/README.md" in release_files(package)


def test_overwrite_regenerates_readme_and_drops_old_notes(
    tmp_path: Path,
) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("Old release description")
    save_catalog(tmp_path, None, "--release-notes", str(notes))

    result = save(tmp_path, "--overwrite")

    assert result.exit_code == 0, result.output
    text = (tmp_path / "snapshot/README.md").read_text()
    assert "Old release description" not in text
    assert "## Release notes" not in text


def test_restored_readme_is_not_copied_into_new_output(
    tmp_path: Path,
) -> None:
    save_catalog(tmp_path)
    destination = tmp_path / "restored"
    restore_snapshot(tmp_path / "snapshot", destination)
    assert (destination / "README.md").is_file()
    # A restored root must not nest the previous release's summary in output/.
    result = runner.invoke(
        app,
        ["save", str(tmp_path / "plain"), "--source-root", str(destination)],
    )
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "plain/output/README.md").exists()
    assert not (tmp_path / "plain/README.md").exists()


def test_readme_conflict_is_refused_before_copying(tmp_path: Path) -> None:
    write_catalog(tmp_path / "source", Catalog(methods=[ALPHA]))
    package = tmp_path / "snapshot"
    package.mkdir()
    (package / "README.md").write_text("Do not replace")

    result = save(tmp_path)

    assert isinstance(result.exception, FileExistsError)
    assert (package / "README.md").read_text() == "Do not replace"
    assert not (package / "output").exists()


@pytest.mark.parametrize("with_release", [False, True])
def test_release_notes_require_existing_file_and_release_id(
    tmp_path: Path,
    with_release: bool,
) -> None:
    notes = tmp_path / "notes.md"
    if not with_release:
        notes.write_text("Description")
    args = ["save", str(tmp_path / "snapshot"), "--release-notes", str(notes)]
    if with_release:
        args.extend(["--release-id", "example"])

    result = runner.invoke(app, args)

    assert result.exit_code != 0
    assert "--release-notes" in result.output
    assert not (tmp_path / "snapshot").exists()
