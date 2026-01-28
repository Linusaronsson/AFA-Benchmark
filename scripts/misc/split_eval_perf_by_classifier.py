import argparse
from pathlib import Path

import polars as pl


def _write_csv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split eval performance results by classifier type."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_builtin", type=Path, required=True)
    parser.add_argument("--output_external", type=Path, required=True)
    args = parser.parse_args()

    df = pl.read_csv(args.input_path, null_values=["null"])
    if "classifier" not in df.columns:
        msg = "Expected 'classifier' column in eval performance CSV."
        raise ValueError(msg)

    builtin = df.filter(pl.col("classifier") == "builtin").drop("classifier")
    external = df.filter(pl.col("classifier") == "external").drop("classifier")

    _write_csv(builtin, args.output_builtin)
    _write_csv(external, args.output_external)


if __name__ == "__main__":
    main()
