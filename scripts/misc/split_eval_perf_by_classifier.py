import argparse
from pathlib import Path

import pandas as pd


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split eval performance results by classifier type."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_builtin", type=Path, required=True)
    parser.add_argument("--output_external", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input_path, dtype_backend="numpy_nullable")
    if "classifier" not in df.columns:
        msg = "Expected 'classifier' column in eval performance dataframe."
        raise ValueError(msg)

    builtin = df.loc[df["classifier"] == "builtin"].drop(columns="classifier")
    external = df.loc[df["classifier"] == "external"].drop(
        columns="classifier"
    )

    _write_parquet(builtin, args.output_builtin)
    _write_parquet(external, args.output_external)


if __name__ == "__main__":
    main()
