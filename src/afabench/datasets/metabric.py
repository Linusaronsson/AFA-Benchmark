"""
METABRIC breast tumours, classified into their molecular subtype from 11 genes.

The features and label are those of the L2M paper (arXiv:2510.12624v2,
Appendix A.7.1); `docs/reference/datasets/metabric.md` describes the source,
its license and how AFABench departs from the paper.
"""

import hashlib
import io
import zipfile
from pathlib import Path
from typing import ClassVar, final, override
from urllib.request import urlopen

import pandas as pd
import torch

from afabench.datasets.tabular_csv import TabularCSVDataset


class MetabricSourceError(ValueError):
    """The METABRIC source file is not the one the dataset is defined on."""


@final
class MetabricDataset(TabularCSVDataset):
    """
    METABRIC restricted to the 11 genes of the L2M paper, labelled with the PAM50 + claudin-low subtype.

    The source is Kaggle's copy of cBioPortal's METABRIC study, downloaded
    when missing and checked against `sha256` on every generation, so a
    generation index names the same tumour in every checkout. The 6 tumours
    of subtype NC (not classified) are dropped; `get_source_rows` maps
    instances back to the file.
    """

    config_keys: ClassVar[tuple[str, ...]] = ("root", "sha256")

    URL: ClassVar[str] = (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "raghadalharbi/breast-cancer-gene-expression-profiles-metabric"
    )
    SOURCE_FILE: ClassVar[str] = "METABRIC_RNA_Mutation.csv"
    # Version 1 of the Kaggle dataset, the only one (2020-05-26)
    SHA256: ClassVar[str] = (
        "d4802594827667a71a75b5ae5c3cad1eb4346222052fcc792361c64abc7e43bc"
    )
    # mRNA z-scores, in the order of both the paper and the source file
    FEATURE_COLUMNS: ClassVar[tuple[str, ...]] = (
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
    )
    LABEL_COLUMN: ClassVar[str] = "pam50_+_claudin-low_subtype"
    # Luminal A, Luminal B, HER2-enriched, basal-like, normal-like and
    # claudin-low, in the paper's order
    CLASSES: ClassVar[tuple[str, ...]] = (
        "LumA",
        "LumB",
        "Her2",
        "Basal",
        "Normal",
        "claudin-low",
    )
    NOT_CLASSIFIED: ClassVar[str] = "NC"

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([len(self.FEATURE_COLUMNS)])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([len(self.CLASSES)])

    def __init__(
        self,
        root: str = "data/metabric/METABRIC_RNA_Mutation.csv",
        sha256: str = SHA256,
    ):
        self.root = root
        self.sha256 = sha256
        super().__init__(Path(root))

    @override
    def _fetch_source(self) -> None:
        with urlopen(self.URL, timeout=60) as response:  # noqa: S310
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        if self.SOURCE_FILE not in archive.namelist():
            msg = (
                f"The archive downloaded from {self.URL} has no "
                f"{self.SOURCE_FILE}, only {archive.namelist()}"
            )
            raise MetabricSourceError(msg)
        content = archive.read(self.SOURCE_FILE)
        # Checked before writing, so a changed upstream file is not kept
        self._check_sha256(content, self.URL)
        self.source.parent.mkdir(parents=True, exist_ok=True)
        partial = self.source.with_name(self.source.name + ".partial")
        partial.write_bytes(content)
        partial.replace(self.source)

    def _check_sha256(self, content: bytes, origin: object) -> None:
        sha256 = hashlib.sha256(content).hexdigest()
        if sha256 != self.sha256:
            msg = (
                f"METABRIC source from {origin} has SHA-256 {sha256}, "
                f"expected {self.sha256}; remove it to download the "
                "expected file again"
            )
            raise MetabricSourceError(msg)

    @override
    def _read_source(self) -> tuple[pd.DataFrame, pd.Series]:
        content = self.source.read_bytes()
        self._check_sha256(content, self.source)
        frame = pd.read_csv(io.BytesIO(content), low_memory=False)
        columns = [*self.FEATURE_COLUMNS, self.LABEL_COLUMN]
        missing = [column for column in columns if column not in frame]
        if missing:
            msg = f"METABRIC source {self.source} has no columns {missing}"
            raise MetabricSourceError(msg)
        frame = frame.loc[frame[self.LABEL_COLUMN] != self.NOT_CLASSIFIED]
        # The base would fill missing values with column means
        with_missing = [c for c in columns if bool(frame[c].isna().any())]
        if with_missing:
            msg = (
                f"METABRIC source {self.source} has missing values in "
                f"{with_missing}"
            )
            raise MetabricSourceError(msg)
        unknown = sorted(
            set(frame[self.LABEL_COLUMN]) - set(self.CLASSES), key=str
        )
        if unknown:
            msg = (
                f"METABRIC source {self.source} has subtypes {unknown}, "
                f"expected {[*self.CLASSES, self.NOT_CLASSIFIED]}"
            )
            raise MetabricSourceError(msg)
        labels = frame[self.LABEL_COLUMN].map(
            {name: index for index, name in enumerate(self.CLASSES)}
        )
        return frame.loc[:, list(self.FEATURE_COLUMNS)].copy(), labels
