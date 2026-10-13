"""METABRIC keeps the L2M paper's 11 genes and six subtypes of a pinned source file."""

import hashlib
import io
import re
import zipfile
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import pytest
import torch

from afabench.core.bundle_system.bundle import load_bundle, save_bundle
from afabench.datasets import MetabricDataset
from afabench.datasets import metabric as metabric_module
from afabench.datasets.metabric import MetabricSourceError
from afabench.testing.provenance import placeholder_provenance

PAPER_GENES = [
    "ccnb1",
    "cdk1",
    "e2f2",
    "e2f7",
    "stat5b",
    "notch1",
    "rbpj",
    "bcl2",
    "egfr",
    "erbb2",
    "erbb3",
]
SUBTYPE = "pam50_+_claudin-low_subtype"
# Row 2 is not classified, so instances 0..6 are rows 0, 1, 3..7
SUBTYPES = ["Normal", "LumA", "NC", "claudin-low", "Basal", "Her2", "LumB"]
SUBTYPES += ["LumA"]


def source_frame() -> pd.DataFrame:
    """Return a small source in the Kaggle file's layout, its genes in reverse order."""
    rng = np.random.default_rng(0)
    n_rows = len(SUBTYPES)
    frame = pd.DataFrame({"patient_id": range(n_rows)})
    frame["age_at_diagnosis"] = rng.uniform(30, 90, n_rows)
    frame[SUBTYPE] = SUBTYPES
    for gene in [*reversed(PAPER_GENES), "abcb1"]:
        frame[gene] = rng.normal(size=n_rows)
    frame["tp53_mut"] = "0"
    return frame


def write_source(path: Path, frame: pd.DataFrame) -> str:
    """Write `frame` as the source; return its SHA-256."""
    frame.to_csv(path, index=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metabric(tmp_path: Path, frame: pd.DataFrame) -> MetabricDataset:
    path = tmp_path / "METABRIC_RNA_Mutation.csv"
    return MetabricDataset(root=str(path), sha256=write_source(path, frame))


def test_features_are_the_paper_genes_in_the_paper_order(
    tmp_path: Path,
) -> None:
    frame = source_frame()

    dataset = metabric(tmp_path, frame)

    assert dataset.feature_names == PAPER_GENES
    kept = frame[frame[SUBTYPE] != "NC"]
    # Normalization keeps each gene's ordering across instances
    for column, gene in enumerate(PAPER_GENES):
        assert dataset.features[:, column].argsort().tolist() == (
            kept[gene].to_numpy().argsort().tolist()
        )


def test_subtypes_are_encoded_in_the_paper_order(tmp_path: Path) -> None:
    dataset = metabric(tmp_path, source_frame())

    # LumA, LumB, Her2, Basal, Normal, claudin-low
    assert dataset.labels.argmax(-1).tolist() == [4, 0, 5, 3, 2, 1, 0]
    assert dataset.label_shape == torch.Size([6])


def test_not_classified_tumours_are_dropped_keeping_source_rows(
    tmp_path: Path,
) -> None:
    dataset = metabric(tmp_path, source_frame())

    assert len(dataset) == 7
    assert dataset.get_source_rows().tolist() == [0, 1, 3, 4, 5, 6, 7]
    assert dataset.get_generation_indices().tolist() == list(range(7))


def test_generation_is_repeatable(tmp_path: Path) -> None:
    first = metabric(tmp_path, source_frame())
    second = MetabricDataset(root=first.root, sha256=first.sha256)

    assert torch.equal(first.features, second.features)
    assert torch.equal(first.labels, second.labels)


def test_bundle_round_trip_keeps_instances_and_config(tmp_path: Path) -> None:
    dataset = metabric(tmp_path, source_frame()).create_subset([5, 0, 2])
    bundle_path = tmp_path / "metabric.bundle"

    save_bundle(
        dataset, bundle_path, metadata={}, provenance=placeholder_provenance()
    )
    loaded = load_bundle(bundle_path).obj

    assert isinstance(loaded, MetabricDataset)
    assert torch.equal(loaded.features, dataset.features)
    assert torch.equal(loaded.labels, dataset.labels)
    assert loaded.get_source_rows().tolist() == [6, 0, 3]
    assert loaded.feature_names == PAPER_GENES
    assert loaded.sha256 == dataset.sha256


def test_source_with_another_checksum_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "METABRIC_RNA_Mutation.csv"
    write_source(path, source_frame())

    with pytest.raises(
        MetabricSourceError, match=r"SHA-256 [0-9a-f]{64}, expected 0{64}"
    ):
        MetabricDataset(root=str(path), sha256="0" * 64)


@pytest.mark.parametrize("column", ["egfr", SUBTYPE])
def test_source_without_a_needed_column_is_rejected(
    tmp_path: Path, column: str
) -> None:
    frame = source_frame().drop(columns=[column])

    with pytest.raises(
        MetabricSourceError, match=rf"no columns \['{re.escape(column)}'"
    ):
        metabric(tmp_path, frame)


@pytest.mark.parametrize("column", ["erbb2", SUBTYPE])
def test_missing_value_in_a_kept_tumour_is_rejected(
    tmp_path: Path, column: str
) -> None:
    frame = source_frame()
    frame.loc[4, column] = None

    with pytest.raises(
        MetabricSourceError, match=rf"missing values.*{re.escape(column)}"
    ):
        metabric(tmp_path, frame)


def test_unknown_subtype_is_rejected(tmp_path: Path) -> None:
    frame = source_frame()
    frame.loc[5, SUBTYPE] = "Her2-enriched"

    with pytest.raises(MetabricSourceError, match=r"\['Her2-enriched'\]"):
        metabric(tmp_path, frame)


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self) -> bytes:
        return self.content


def serve_archive(
    monkeypatch: pytest.MonkeyPatch, members: dict[str, bytes]
) -> list[str]:
    """Make the download return a zip of `members`; return the URLs requested."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    requested: list[str] = []

    def fake_urlopen(url: str, timeout: float) -> FakeResponse:  # noqa: ARG001
        requested.append(url)
        return FakeResponse(buffer.getvalue())

    monkeypatch.setattr(metabric_module, "urlopen", fake_urlopen)
    return requested


def source_bytes() -> bytes:
    return source_frame().to_csv(index=False).encode()


def test_missing_source_is_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = source_bytes()
    requested = serve_archive(
        monkeypatch, {"METABRIC_RNA_Mutation.csv": content}
    )
    path = tmp_path / "metabric" / "METABRIC_RNA_Mutation.csv"

    dataset = MetabricDataset(
        root=str(path), sha256=hashlib.sha256(content).hexdigest()
    )

    assert requested == [MetabricDataset.URL]
    assert path.read_bytes() == content
    assert len(dataset) == 7


def test_present_source_is_not_downloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = serve_archive(monkeypatch, {})

    metabric(tmp_path, source_frame())

    assert requested == []


def test_downloaded_source_with_another_checksum_is_not_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_archive(monkeypatch, {"METABRIC_RNA_Mutation.csv": source_bytes()})
    path = tmp_path / "METABRIC_RNA_Mutation.csv"

    with pytest.raises(MetabricSourceError, match="SHA-256"):
        MetabricDataset(root=str(path), sha256="0" * 64)
    assert list(tmp_path.iterdir()) == []


def test_archive_without_the_source_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serve_archive(monkeypatch, {"other.csv": b"a,b\n"})

    with pytest.raises(MetabricSourceError, match=r"no METABRIC_RNA.*other"):
        MetabricDataset(root=str(tmp_path / "METABRIC_RNA_Mutation.csv"))


@pytest.mark.optional
def test_real_source_has_the_paper_cohort() -> None:
    """Downloads the source into data/metabric/ if it is not there."""
    dataset = MetabricDataset()

    assert len(dataset) == 1898
    assert dataset.feature_names == PAPER_GENES
    counts = dataset.labels.sum(0).int().tolist()
    # LumA, LumB, Her2, Basal, Normal, claudin-low
    assert counts == [679, 461, 220, 199, 140, 199]
