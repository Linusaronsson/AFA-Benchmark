"""
Add dataset_split column to CSV file.

Replaces scripts/misc/add_dataset_split_col.R with pure Python.

Usage:
    python add_dataset_split_col.py input.csv output.csv 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    """Run the main entry point."""
    parser = argparse.ArgumentParser(
        description="Add dataset_split column to CSV"
    )
    parser.add_argument("input_path", type=Path, help="Input CSV path")
    parser.add_argument("output_path", type=Path, help="Output CSV path")
    parser.add_argument("dataset_split", type=int, help="Dataset split value")

    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    df["dataset_split"] = args.dataset_split

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
