#!/usr/bin/env python3
import argparse

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Parquet files with union of columns."
    )
    parser.add_argument("inputs", nargs="*", help="Input Parquet files")
    parser.add_argument("--output", required=True, help="Output Parquet file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dfs = []
    for fp in args.inputs:
        df = pl.read_parquet(fp)
        dfs.append(df)

    df_merged = pl.concat(dfs, how="vertical", rechunk=True)
    df_merged.write_parquet(args.output)


if __name__ == "__main__":
    main()
