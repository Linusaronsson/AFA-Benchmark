from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge eval-performance parquets across comparison conditions "
            "and annotate each input with a condition label."
        )
    )
    parser.add_argument(
        "--input",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
        help="Comparison label and parquet path.",
    )
    parser.add_argument(
        "--group-column",
        type=str,
        default="comparison_group",
        help="Column name used for the comparison label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataframes: list[pl.DataFrame] = []
    for label, path_str in args.input:
        path = Path(path_str)
        if not path.exists():
            msg = f"Input parquet does not exist: {path}"
            raise FileNotFoundError(msg)

        dataframe = pl.read_parquet(path).with_columns(
            pl.lit(label, dtype=pl.String).alias(args.group_column)
        )
        dataframes.append(dataframe)

    if not dataframes:
        msg = "Expected at least one input parquet."
        raise ValueError(msg)

    merged = pl.concat(dataframes, how="diagonal_relaxed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(args.output)


if __name__ == "__main__":
    main()
