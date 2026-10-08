"""
The job duration table: every job record of a pipeline's outputs as one row.

Design: `docs/adr/0006-job-records-beside-artifacts.md`; columns:
`docs/reference/job_records.md`. Rows are not aggregated: each job record,
completed, failed or timed out, is one row, so repeated attempts of one job
are separate rows. `load_job_duration_table` reads either an output root of
loose job records or a job duration table written from one, and returns the
same frame for both.
"""

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from afabench.core.job_record import (
    JOB_RECORD_SUFFIX,
    JOB_RECORD_VERSION,
)

KNOWN_JOB_RECORD_VERSIONS = {JOB_RECORD_VERSION}

# One column per field of a job record (`JobRecord`). Nullable dtypes, since
# null means unknown.
RECORD_FIELD_DTYPES = {
    "job_record_version": "Int64",
    "stage": "string",
    "name": "string",
    "dataset_key": "string",
    "dataset_realization_index": "Int64",
    "pretrain_seed": "Int64",
    "train_seed": "Int64",
    "eval_seed": "Int64",
    "train_hard_budget": "Int64",
    "train_soft_budget_param": "Float64",
    "eval_hard_budget": "Int64",
    "eval_soft_budget_param": "Float64",
    "eval_batch_size": "Int64",
    "started_at": "datetime64[us, UTC]",
    "ended_at": "datetime64[us, UTC]",
    "job_duration_seconds": "Float64",
    "exit_status": "string",
    "exit_code": "Int64",
    "device": "string",
    "cpus": "Int64",
    "gpus": "Int64",
    "time_limit_minutes": "Int64",
    "gpu_model": "string",
    "cpu_model": "string",
    "host": "string",
    "slurm_job_id": "string",
    "code_commit": "string",
    "smoke_test": "boolean",
}
# The path of the row's job record relative to the output root comes first.
COLUMN_DTYPES = {"job_record_path": "string", **RECORD_FIELD_DTYPES}


class UnknownJobRecordVersionError(ValueError):
    """A job record, or a table row, has a schema version this code lacks."""


class JobRecordFieldsError(ValueError):
    """A job record, or a table, lacks or adds fields of its version."""


def load_job_duration_table(source: Path) -> pd.DataFrame:
    """
    Load a job duration table, or the job records under an output root.

    `source` is a table's Parquet file or an output root directory. Rows
    are sorted by `job_record_path`.
    """
    if not source.is_dir():
        table = pd.read_parquet(source)
        for version in table["job_record_version"].unique():
            _check_version(version, source)
        _check_fields(table.columns, COLUMN_DTYPES, source)
        return _typed(table.reindex(columns=pd.Index(COLUMN_DTYPES)))
    rows = [
        {
            "job_record_path": path.relative_to(source).as_posix(),
            **_read_job_record(path),
        }
        for path in source.rglob(f"*{JOB_RECORD_SUFFIX}")
    ]
    return _typed(pd.DataFrame(rows, columns=pd.Index(COLUMN_DTYPES)))


def empty_job_duration_table() -> pd.DataFrame:
    """Return a job duration table without rows, as of an empty output root."""
    return _typed(pd.DataFrame(columns=pd.Index(COLUMN_DTYPES)))


def write_job_duration_table(output_root: Path, table_path: Path) -> None:
    """Write the job duration table of an output root's job records."""
    table_path.parent.mkdir(parents=True, exist_ok=True)
    load_job_duration_table(output_root).to_parquet(table_path, index=False)


def _read_job_record(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text())
    _check_version(record.get("job_record_version"), path)
    _check_fields(record, RECORD_FIELD_DTYPES, path)
    return record


def _check_version(version: object, source: Path) -> None:
    if version not in KNOWN_JOB_RECORD_VERSIONS:
        message = (
            f"Unknown job record version {version} in {source}; known "
            f"versions: {sorted(KNOWN_JOB_RECORD_VERSIONS)}"
        )
        raise UnknownJobRecordVersionError(message)


def _check_fields(
    present: Iterable[str], expected: Iterable[str], source: Path
) -> None:
    missing = set(expected) - set(present)
    unexpected = set(present) - set(expected)
    if missing or unexpected:
        message = (
            f"Missing job record fields {sorted(missing)} and unexpected "
            f"fields {sorted(unexpected)} in {source}"
        )
        raise JobRecordFieldsError(message)


def _typed(table: pd.DataFrame) -> pd.DataFrame:
    table = table.astype(
        {
            column: dtype
            for column, dtype in COLUMN_DTYPES.items()
            if not dtype.startswith("datetime")
        }
    )
    for column in ["started_at", "ended_at"]:
        table[column] = pd.to_datetime(table[column], utc=True).astype(
            COLUMN_DTYPES[column]
        )
    return table.sort_values("job_record_path", ignore_index=True)
