from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence


def _as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def scan_parquet(path: str | Path) -> pl.LazyFrame:
    """Open a parquet file lazily so downstream filters can stream from disk."""
    parquet_path = _as_path(path)
    if not parquet_path.exists():
        msg = f"Input parquet does not exist: {parquet_path}"
        raise FileNotFoundError(msg)
    return pl.scan_parquet(parquet_path)


def collect_streaming(frame: pl.LazyFrame) -> pl.DataFrame:
    """Collect a lazy frame using Polars' streaming engine."""
    return frame.collect(engine="streaming")


def sink_parquet(frame: pl.LazyFrame, path: str | Path) -> None:
    """Write a lazy parquet plan without materializing the full frame in Python."""
    output_path = _as_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.sink_parquet(output_path)


def require_columns(
    frame: pl.LazyFrame,
    required_columns: Iterable[str],
) -> pl.Schema:
    """Validate that a parquet-backed lazy frame contains the required columns."""
    schema = frame.collect_schema()
    missing_columns = sorted(set(required_columns).difference(schema.names()))
    if missing_columns:
        msg = f"Input parquet missing required columns: {missing_columns}"
        raise ValueError(msg)
    return schema


def split_parquet_by_column_values(
    input_path: str | Path,
    *,
    column: str,
    outputs_by_value: Mapping[Any, str | Path],
    drop_column: bool = False,
) -> None:
    """Split a parquet file into multiple outputs by a categorical column."""
    frame = scan_parquet(input_path)
    schema = require_columns(frame, [column])
    selected_columns = [
        name for name in schema.names() if not (drop_column and name == column)
    ]

    for value, output_path in outputs_by_value.items():
        sink_parquet(
            frame.filter(pl.col(column) == pl.lit(value)).select(
                selected_columns
            ),
            output_path,
        )


def concat_labeled_parquets(
    inputs: Sequence[tuple[str, str | Path]],
    *,
    label_column: str,
    output_path: str | Path,
) -> None:
    """Concatenate parquet inputs lazily and annotate each with a label column."""
    if not inputs:
        msg = "Expected at least one input parquet."
        raise ValueError(msg)

    frames = [
        scan_parquet(path).with_columns(
            pl.lit(label, dtype=pl.String).alias(label_column)
        )
        for label, path in inputs
    ]
    sink_parquet(pl.concat(frames, how="diagonal_relaxed"), output_path)
