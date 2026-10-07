"""The provenance record and its capture (ADR 0002, required tests 4 and 5)."""

import os
import subprocess
from pathlib import Path

import pytest

from afabench.core.provenance import (
    PROVENANCE_VERSION,
    DatasetIdentity,
    DatasetIdentityMismatchError,
    ProvenanceInput,
    ProvenanceRecord,
    UnknownProvenanceVersionError,
    capture_provenance,
    shared_dataset_identity,
)


def _git(repository: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repository), *args],  # noqa: S607
        check=True,
        capture_output=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "HOME": str(repository),
        },
    )


def _capture(**overrides: object) -> ProvenanceRecord:
    arguments: dict[str, object] = {
        "stage": "classifier_training",
        "resolved_config": {"lr": 0.1, "layers": [8, 8], "nested": {"a": 1}},
        "seed": 3,
        "smoke_test": True,
        "device": "cpu",
        "inputs": [
            ProvenanceInput(
                role="train_dataset",
                path="datasets/cube/0/train.bundle",
                class_name="CubeDataset",
                content_hash="sha256:" + "0" * 64,
            ),
            ProvenanceInput(
                role="val_dataset",
                path="datasets/cube/0/val.bundle",
                class_name="CubeDataset",
                content_hash=None,
            ),
        ],
        "dataset_key": "cube",
        "dataset_realization_index": 0,
    }
    arguments.update(overrides)
    return capture_provenance(**arguments)  # pyright: ignore[reportArgumentType]


def test_record_survives_a_json_round_trip() -> None:
    record = _capture()

    assert ProvenanceRecord.from_json_dict(record.to_json_dict()) == record
    assert record.provenance_version == PROVENANCE_VERSION
    assert record.stage == "classifier_training"
    assert record.seed == 3
    assert record.smoke_test is True
    assert record.method_name is None
    assert record.split is None
    assert record.dataset_key == "cube"
    assert record.dataset_realization_index == 0
    assert record.resolved_config == {
        "lr": 0.1,
        "layers": [8, 8],
        "nested": {"a": 1},
    }
    assert record.environment.python_version.startswith("3.12")
    assert record.compute.device == "cpu"
    assert record.compute.accelerator_name is None


def test_reader_rejects_an_unknown_provenance_version() -> None:
    serialised = _capture().to_json_dict()
    serialised["provenance_version"] = 99

    with pytest.raises(UnknownProvenanceVersionError, match="99"):
        ProvenanceRecord.from_json_dict(serialised)


def test_capture_rejects_a_config_value_that_is_not_json() -> None:
    with pytest.raises(TypeError, match=r"optimizer\.betas.*\(0\.9, 0\.99\)"):
        _capture(resolved_config={"optimizer": {"betas": (0.9, 0.99)}})


def test_disagreeing_dataset_identities_raise_naming_both_values() -> None:
    train = _capture(dataset_key="cube", dataset_realization_index=0)
    val = _capture(dataset_key="cube", dataset_realization_index=1)

    with pytest.raises(DatasetIdentityMismatchError, match=r"0.*1"):
        shared_dataset_identity(train, val)
    with pytest.raises(DatasetIdentityMismatchError, match=r"cube.*mnist"):
        shared_dataset_identity(train, _capture(dataset_key="mnist"))


def test_agreeing_dataset_identities_are_copied() -> None:
    train = _capture(dataset_key="cube", dataset_realization_index=2)
    val = _capture(dataset_key="cube", dataset_realization_index=2)

    assert shared_dataset_identity(train, val) == DatasetIdentity(
        dataset_key="cube", dataset_realization_index=2
    )


def test_identity_of_inputs_without_a_record_is_unknown() -> None:
    assert shared_dataset_identity(None, None) == DatasetIdentity(
        dataset_key=None, dataset_realization_index=None
    )
    assert shared_dataset_identity(
        None, _capture(dataset_key="cube", dataset_realization_index=4)
    ) == DatasetIdentity(dataset_key="cube", dataset_realization_index=4)


def test_capture_outside_a_git_work_tree_records_unknown_code(
    tmp_path: Path,
) -> None:
    record = _capture(checkout=tmp_path)

    assert record.code_commit is None
    assert record.code_dirty is None


def test_capture_run_from_outside_the_checkout_records_afabench_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).parents[4]
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.chdir(tmp_path)

    record = _capture()

    assert record.code_commit == commit
    assert record.environment.lockfile_sha256 is not None


def test_capture_in_a_repository_records_commit_and_dirty_flag(
    tmp_path: Path,
) -> None:
    tracked = tmp_path / "tracked.py"
    tracked.write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-q", "-m", "init")
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Untracked files (outputs, data) do not make the code dirty
    (tmp_path / "untracked.txt").write_text("ignored\n")
    clean = _capture(checkout=tmp_path)
    tracked.write_text("x = 2\n")
    dirty = _capture(checkout=tmp_path)

    assert (clean.code_commit, clean.code_dirty) == (commit, False)
    assert (dirty.code_commit, dirty.code_dirty) == (commit, True)
