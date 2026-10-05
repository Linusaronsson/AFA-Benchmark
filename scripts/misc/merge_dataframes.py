#!/usr/bin/env python3
import argparse
from collections import OrderedDict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# WARNING: LLM generated, supposedly fixes OOM errors compared to
# concatenating all inputs in memory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Parquet files with union of columns (streaming, low mem)."
    )
    parser.add_argument("inputs", nargs="*", help="Input Parquet files")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    return parser.parse_args()


def is_any_string_type(typ: pa.DataType) -> bool:
    return (
        pa.types.is_string(typ)
        or pa.types.is_large_string(typ)
        or pa.types.is_binary(typ)
        or pa.types.is_fixed_size_binary(typ)
        or pa.types.is_large_binary(typ)
    )


def get_schema_union(parquet_files: list[str]) -> pa.Schema:
    # Build union of all schemas, upcasting to string if needed

    all_columns = OrderedDict()
    for path in parquet_files:
        if not Path(path).exists():
            msg = f"Input file does not exist: {path}"
            raise FileNotFoundError(msg)
        if Path(path).stat().st_size == 0:
            msg = f"Input file is empty: {path}"
            raise ValueError(msg)

        schema = pq.read_schema(path)
        for name, f in zip(schema.names, schema, strict=True):
            if name not in all_columns:
                # Normalize all string types to pa.string()
                typ = f.type
                if is_any_string_type(typ):
                    all_columns[name] = pa.string()
                else:
                    all_columns[name] = typ
            else:
                existing = all_columns[name]
                this = f.type
                if existing == this:
                    continue
                # if either is string-like or binary, promote to pa.string()
                if is_any_string_type(existing) or is_any_string_type(this):
                    all_columns[name] = pa.string()
                else:
                    # fallback: upcast to pa.string()
                    all_columns[name] = pa.string()

    if not all_columns:
        msg = "No readable input files provided"
        raise ValueError(msg)

    names = list(all_columns.keys())
    types = [all_columns[k] for k in names]
    return pa.schema([(k, t) for k, t in zip(names, types, strict=True)])


def _process_file(
    path: str, full_schema: pa.Schema, writer: pq.ParquetWriter
) -> None:
    """
    Process a single parquet file and write aligned rows to writer.

    Raises an exception if the file cannot be processed.
    """
    if not Path(path).exists():
        msg = f"Input file does not exist: {path}"
        raise FileNotFoundError(msg)
    if Path(path).stat().st_size == 0:
        msg = f"Input file is empty: {path}"
        raise ValueError(msg)

    table = pq.read_table(path)

    if table.num_rows == 0 or table.num_columns == 0:
        msg = f"DataFrame in {path} is empty"
        raise ValueError(msg)

    # Align to union schema, in strict column order: fill missing columns
    # with nulls and cast columns that were promoted (to string) in the union
    columns = []
    for field in full_schema:
        if field.name not in table.column_names:
            columns.append(pa.nulls(table.num_rows, type=field.type))
        else:
            columns.append(pc.cast(table[field.name], field.type))

    writer.write_table(pa.table(columns, schema=full_schema))


def main() -> None:
    args = parse_args()
    files = args.inputs
    output_path = args.output

    if not files:
        msg = "No input files provided"
        raise ValueError(msg)

    # 1. Union schema discovery
    full_schema = get_schema_union(files)

    writer = pq.ParquetWriter(output_path, full_schema)
    try:
        for path in files:
            _process_file(path, full_schema, writer)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
