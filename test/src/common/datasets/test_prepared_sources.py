"""Tabular datasets read a prepared source pinned by SHA-256, which source preparation rebuilds from the provider's raw files (ADR 0014)."""

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from afabench.datasets import (
    ACTG175Dataset,
    CKDDataset,
    DiabetesDataset,
    MiniBooNEDataset,
    MiniBooNESubsetDataset,
    PhysionetChallenge2012Dataset,
    tabular_csv,
)
from afabench.datasets.tabular_csv import (
    NoSourcePreparationError,
    SourceChecksumError,
)
from scripts.dataset_generation.prepare_source import REPO_ROOT, source_of


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def uci_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(6, 3)), columns=["a", "b", "c"])
    frame["target"] = [0, 1, 0, 1, 1, 0]
    return frame


def serve(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> None:
    """Make the download return `files[name]` for a URL ending in `name`."""
    monkeypatch.setattr(
        tabular_csv, "download", lambda url: files[url.rsplit("/", 1)[-1]]
    )


def test_missing_committed_source_names_its_preparation(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match=r"prepare_source\.py"):
        CKDDataset(path=str(tmp_path / "ckd.csv"))


def test_changed_committed_source_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ckd.csv"
    uci_frame().to_csv(path, index=False)

    with pytest.raises(
        SourceChecksumError, match=r"expected 0{64}; restore the committed"
    ):
        CKDDataset(path=str(path), sha256="0" * 64)


def test_source_kept_out_of_the_repository_is_prepared_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ACTG175Dataset, "_fetch_uci", classmethod(lambda _: uci_frame())
    )
    content = uci_frame().to_csv(index=False).encode()
    path = tmp_path / "actg" / "actg.csv"

    dataset = ACTG175Dataset(path=str(path), sha256=sha256(content))

    assert path.read_bytes() == content
    assert len(dataset) == 6


def test_prepared_source_with_another_checksum_is_not_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        CKDDataset, "_fetch_uci", classmethod(lambda _: uci_frame())
    )

    with pytest.raises(SourceChecksumError, match=r"expected 0{64}"):
        CKDDataset.prepare_source(tmp_path / "ckd.csv", "0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_source_without_preparation_says_so(tmp_path: Path) -> None:
    with pytest.raises(NoSourcePreparationError, match="committed"):
        DiabetesDataset.prepare_source(tmp_path / "diabetes.csv")


def miniboone_archive() -> bytes:
    """Return a UCI-style archive of 2 signal and 3 background events, each feature its event's number."""
    lines = [" 2 3"]
    lines += [
        "  " + "  ".join(f"{event:.6E}" for _ in range(50))
        for event in range(5)
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("MiniBooNE_PID.txt", "\n".join(lines) + "\n")
    return buffer.getvalue()


@pytest.mark.parametrize(
    "dataset_class", [MiniBooNEDataset, MiniBooNESubsetDataset]
)
def test_miniboone_source_lists_signal_then_background_events(
    monkeypatch: pytest.MonkeyPatch,
    dataset_class: type[MiniBooNEDataset | MiniBooNESubsetDataset],
) -> None:
    serve(
        monkeypatch,
        {MiniBooNEDataset.URL.rsplit("/", 1)[-1]: miniboone_archive()},
    )

    frame = pd.read_csv(io.BytesIO(dataset_class.make_prepared_source()))

    assert list(frame.columns) == [
        *(f"Feature {i}" for i in range(1, 51)),
        "Outcome",
    ]
    assert frame["Feature 50"].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert frame["Outcome"].tolist() == [1, 1, 0, 0, 0]


def record(record_id: int, rows: list[tuple[str, str, float]]) -> bytes:
    lines = ["Time,Parameter,Value", f"00:00,RecordID,{record_id}"]
    lines += [f"{time},{name},{value}" for time, name, value in rows]
    return ("\n".join(lines) + "\n").encode()


def physionet_files() -> dict[str, bytes]:
    """Return raw Challenge files of two stays per set, set b's listed out of order in its archive."""
    stays = {
        "a": {
            "132539": [("00:00", "Age", 54), ("01:00", "HR", 70)],
            "132540": [("00:00", "Age", 76), ("02:00", "HR", 80)],
        },
        "b": {
            "142999": [("00:00", "Age", 40), ("03:00", "HR", -1)],
            "142671": [("00:00", "Age", 60), ("05:00", "HR", 90)],
        },
        "c": {
            "152871": [("00:00", "Age", 30), ("01:00", "HR", 60)],
            "152872": [
                ("00:00", "Age", 20),
                ("01:00", "HR", 61),
                ("02:00", "HR", 63),
            ],
        },
    }
    deaths = {"132539": 0, "132540": 1, "142999": 0}
    deaths |= {"142671": 1, "152871": 1, "152872": 0}
    files: dict[str, bytes] = {}
    for letter, records in stays.items():
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for record_id, rows in records.items():
                content = record(int(record_id), rows)
                info = tarfile.TarInfo(f"set-{letter}/{record_id}.txt")
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
        files[f"set-{letter}.tar.gz"] = buffer.getvalue()
        outcome_lines = ["RecordID,SAPS-I,In-hospital_death"]
        outcome_lines += [f"{rid},1,{deaths[rid]}" for rid in records]
        files[f"Outcomes-{letter}.txt"] = "\n".join(outcome_lines).encode()
    return files


def test_physionet_source_has_each_stays_parameter_means(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = physionet_files()
    serve(monkeypatch, files)
    monkeypatch.setattr(
        PhysionetChallenge2012Dataset,
        "RAW_SHA256",
        {name: sha256(content) for name, content in files.items()},
    )

    frame = pd.read_csv(
        io.BytesIO(PhysionetChallenge2012Dataset.make_prepared_source())
    )

    assert list(frame.columns) == ["Age", "HR", "Outcome"]
    # Sets a, b and c in order, by record file name within each set
    assert frame["Age"].tolist() == [54, 76, 60, 40, 30, 20]
    # -1 marks a missing value, so stay 142999 has no HR
    assert frame["HR"].isna().tolist() == [False] * 3 + [True] + [False] * 2
    assert frame["HR"].dropna().tolist() == [70, 80, 90, 60, 62]
    assert frame["Outcome"].tolist() == [0, 1, 1, 0, 1, 0]


def test_physionet_raw_file_with_another_checksum_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = physionet_files()
    serve(monkeypatch, files)
    expected = {name: sha256(content) for name, content in files.items()}
    expected["set-b.tar.gz"] = "0" * 64
    monkeypatch.setattr(PhysionetChallenge2012Dataset, "RAW_SHA256", expected)

    with pytest.raises(SourceChecksumError, match=r"set-b\.tar\.gz"):
        PhysionetChallenge2012Dataset.prepare_source(tmp_path / "p.csv")


@pytest.mark.parametrize(
    ("dataset_key", "dataset_class", "folder"),
    [
        ("actg", ACTG175Dataset, "actg"),
        ("ckd", CKDDataset, "ckd"),
        ("diabetes", DiabetesDataset, "diabetes"),
        ("miniboone", MiniBooNEDataset, "miniboone"),
        # Reads its parent's source (ADR 0010)
        ("miniboone_subset", MiniBooNESubsetDataset, "miniboone"),
        (
            "physionet_challenge_2012",
            PhysionetChallenge2012Dataset,
            "physionet_challenge_2012",
        ),
    ],
)
def test_preparation_writes_where_generation_reads(
    dataset_key: str, dataset_class: type, folder: str
) -> None:
    found_class, path, _ = source_of(dataset_key)

    assert found_class is dataset_class
    assert path.parent == REPO_ROOT / "data" / folder


def test_metabric_preparation_uses_the_configs_checksum() -> None:
    _, _, configured = source_of("metabric")

    assert configured is not None
    assert len(configured) == 64


def test_dataset_without_a_tabular_source_has_no_preparation() -> None:
    with pytest.raises(TypeError, match="no tabular source"):
        source_of("cube")
