import argparse
from pathlib import Path

from afabench.common.parquet import split_parquet_by_column_values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split eval performance results by classifier type."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_builtin", type=Path, required=True)
    parser.add_argument("--output_external", type=Path, required=True)
    args = parser.parse_args()

    split_parquet_by_column_values(
        args.input_path,
        column="classifier",
        outputs_by_value={
            "builtin": args.output_builtin,
            "external": args.output_external,
        },
        drop_column=True,
    )


if __name__ == "__main__":
    main()
