from __future__ import annotations

import argparse
from pathlib import Path

from afabench.common.parquet import concat_labeled_parquets


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
    concat_labeled_parquets(
        [(label, Path(path_str)) for label, path_str in args.input],
        label_column=args.group_column,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
