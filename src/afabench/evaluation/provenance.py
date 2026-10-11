"""
The provenance record of an evaluation table, in its Parquet schema metadata.

Design: `docs/adr/0002-provenance-recorded-in-artifacts.md`. The record is
stored as JSON under the Arrow schema metadata key `afabench.provenance`.
pandas drops that metadata on read, so the identity it implies is also
written as columns (`afabench.evaluation.schemas.SavedEvaluationSchema`).
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from afabench.core.provenance import ProvenanceRecord

PROVENANCE_METADATA_KEY = b"afabench.provenance"


def save_evaluation_table(
    frame: pd.DataFrame,
    path: Path,
    *,
    provenance: ProvenanceRecord | None,
) -> None:
    """
    Write `frame` as Parquet with `provenance` in its schema metadata.

    A null record writes no key, for a table transformed from a source table
    written without one.
    """
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if provenance is not None:
        table = table.replace_schema_metadata(
            {
                **(table.schema.metadata or {}),
                PROVENANCE_METADATA_KEY: json.dumps(
                    provenance.to_json_dict()
                ).encode(),
            }
        )
    pq.write_table(table, path)


def evaluation_table_provenance(path: Path) -> ProvenanceRecord | None:
    """Read an evaluation table's record; null for a table without one."""
    metadata = pq.read_schema(path).metadata or {}
    if PROVENANCE_METADATA_KEY not in metadata:
        return None
    return ProvenanceRecord.from_json_dict(
        json.loads(metadata[PROVENANCE_METADATA_KEY])
    )
