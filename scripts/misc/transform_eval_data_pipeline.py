"""
Chain all evaluation data transformations in a single Python script.

This replaces the multi-step nushell approach with pure Python.
"""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "transform_eval_data",
    Path(__file__).parent / "transform_eval_data.py",
)
assert spec is not None
transform_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(transform_module)

NA_REP = transform_module.NA_REP
add_metadata_columns = transform_module.add_metadata_columns
count_selections = transform_module.count_selections
pivot_long_classifier = transform_module.pivot_long_classifier


def transform_eval_data_pipeline(
    input_path: Path,
    output_path: Path,
    method: str,
    dataset: str,
    train_seed: int,
    train_hard_budget: str,
    train_soft_budget_param: str,
) -> None:
    """Apply all transformations in sequence."""
    # Step 1: Load input
    df = pd.read_csv(input_path)

    # Step 2: Add eval metadata
    df = add_metadata_columns(df, {"eval_soft_budget_param": ""})

    # Step 3: Remove selections (count and drop prev_selections_performed)
    df = count_selections(df)

    # Step 4: Add training metadata
    metadata = {
        "afa_method": method,
        "dataset": dataset,
        "train_seed": int(train_seed),
        "train_hard_budget": train_hard_budget
        if train_hard_budget != "null"
        else None,
        "train_soft_budget_param": float(train_soft_budget_param)
        if train_soft_budget_param != "null"
        else None,
    }
    df = add_metadata_columns(df, metadata)

    # Step 5: Pivot classifier columns
    df = pivot_long_classifier(df)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, na_rep=NA_REP)


def main() -> None:
    """Transform evaluation data in a single pipeline."""
    parser = argparse.ArgumentParser(
        description="Transform evaluation data in a single pipeline"
    )
    parser.add_argument("input_path", type=Path, help="Input CSV path")
    parser.add_argument("output_path", type=Path, help="Output CSV path")
    parser.add_argument("method", help="AFA method name")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("train_seed", help="Training seed")
    parser.add_argument("train_hard_budget", help="Training hard budget")
    parser.add_argument(
        "train_soft_budget_param", help="Training soft budget parameter"
    )

    args = parser.parse_args()

    transform_eval_data_pipeline(
        input_path=args.input_path,
        output_path=args.output_path,
        method=args.method,
        dataset=args.dataset,
        train_seed=args.train_seed,
        train_hard_budget=args.train_hard_budget,
        train_soft_budget_param=args.train_soft_budget_param,
    )


if __name__ == "__main__":
    main()
