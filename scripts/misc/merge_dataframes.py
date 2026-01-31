#!/usr/bin/env python3
import argparse
from pathlib import Path
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import sys

# WARNING: LLM generated, supposedly fixes OOM errors compared to normal pl.concat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Parquet files with union of columns (streaming, low mem)."
    )
    parser.add_argument("inputs", nargs="*", help="Input Parquet files")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    return parser.parse_args()


def is_any_string_type(typ):
    return (
        pa.types.is_string(typ)
        or pa.types.is_large_string(typ)
        or pa.types.is_binary(typ)
        or pa.types.is_fixed_size_binary(typ)
        or pa.types.is_large_binary(typ)
    )


def get_schema_union(parquet_files):
    # Build union of all schemas, upcasting to string if needed
    from collections import OrderedDict

    all_columns = OrderedDict()
    for path in parquet_files:
        if not Path(path).exists() or Path(path).stat().st_size == 0:
            continue
        try:
            schema = pq.read_schema(path)
        except Exception as e:
            print(
                f"[WARN] Can't read schema from {path}: {e}", file=sys.stderr
            )
            continue
        for name, f in zip(schema.names, schema):
            if name not in all_columns:
                all_columns[name] = f.type
            else:
                existing = all_columns[name]
                this = f.type
                if existing == this:
                    continue
                # if either is string-like or binary, promote to pa.string()
                if is_any_string_type(existing) or is_any_string_type(this):
                    all_columns[name] = pa.string()
                else:
                    # fallback: upcast to large_string or object
                    all_columns[name] = pa.string()
    names = list(all_columns.keys())
    types = [all_columns[k] for k in names]
    return pa.schema([(k, t) for k, t in zip(names, types)])


def main() -> None:
    args = parse_args()
    files = args.inputs
    output_path = args.output

    # 1. Union schema discovery
    full_schema = get_schema_union(files)
    if len(full_schema) == 0:
        print(
            "[WARN] No readable input, writing empty Parquet file.",
            file=sys.stderr,
        )
        pq.write_table(pa.Table.from_pydict({}), output_path)
        return

    writer = pq.ParquetWriter(output_path, full_schema)
    wrote_any = False
    for path in files:
        if not Path(path).exists() or Path(path).stat().st_size == 0:
            print(
                f"[WARN] Skipping non-existent or empty file: {path}",
                file=sys.stderr,
            )
            continue
        try:
            # Use Polars for convenient reading; convert to Arrow Table
            df = pl.read_parquet(path)
        except Exception as e:
            print(
                f"[WARN] Could not read '{path}': {e}. Skipping.",
                file=sys.stderr,
            )
            continue
        if df.height == 0 or df.width == 0:
            print(
                f"[WARN] Empty DataFrame in {path}; skipping.", file=sys.stderr
            )
            continue
        # Align to union schema
        for name in full_schema.names:
            if name not in df.columns:
                typ = full_schema.field(name).type
                if is_any_string_type(typ):
                    pltype = pl.Utf8
                elif pa.types.is_integer(typ):
                    pltype = pl.Int64
                elif pa.types.is_floating(typ):
                    pltype = pl.Float64
                else:
                    pltype = pl.Object
                df = df.with_columns(pl.lit(None, dtype=pltype).alias(name))
            # Ensure column promoted if type ambiguous: cast to Utf8 if stringy
            elif is_any_string_type(full_schema.field(name).type):
                df = df.with_columns(df[name].cast(pl.Utf8).alias(name))
        # Strict col order
        df = df.select(full_schema.names)
        table = df.to_arrow()
        writer.write_table(table)
        wrote_any = True
    writer.close()
    if not wrote_any:
        print(
            "[WARN] All input files failed/empty; output is empty.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
