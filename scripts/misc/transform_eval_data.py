"""
Transform evaluation data for plotting and analysis.

This script provides CLI commands for various CSV transformations used in
Snakemake workflows, replacing nu shell operations with pure Python.

Usage:
    python transform_eval_data.py count_selections input.csv output.csv
    python transform_eval_data.py add_metadata input.csv output.csv --col key=value
    python transform_eval_data.py pivot_long_classifier input.csv output.csv
    python transform_eval_data.py validate_budgets input.csv output.csv
    python transform_eval_data.py transform input.csv output.csv METHOD [options]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NA_REP = "null"


def nullable_int(value: str) -> int | None:
    """Convert string to int, treating NA_REP as None."""
    if value.lower() == NA_REP:
        return None
    return int(value)


def nullable_float(value: str) -> float | None:
    """Convert string to float, treating NA_REP as None."""
    if value.lower() == NA_REP:
        return None
    return float(value)


def nullable_str(value: str) -> str | None:
    """Convert string to str, treating NA_REP as None."""
    if value.lower() == NA_REP:
        return None
    return value


def is_missing_value(val: object) -> bool:
    if val is None or val is pd.NA:
        return True
    if isinstance(val, float):
        return bool(np.isnan(val))
    return False


def count_selections(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prev_selections_performed JSON list to selections_performed count.

    Replaces nu shell:
        open {input} |
            insert selections_performed {
                get prev_selections_performed | each { |row| ($row | from json | length) + 1}
            } |
            reject prev_selections_performed selection_performed |
                save {output}

    Args:
        df: DataFrame with 'prev_selections_performed' column (JSON string of list)

    Returns:
        DataFrame with 'selections_performed' column, without original columns
    """
    df = df.copy()

    def parse_and_count(x: object) -> int:
        if is_missing_value(x):
            return 1
        if isinstance(x, str):
            if x in {"", "[]"}:
                return 1
            parser = (
                ast.literal_eval if "'" in x and '"' not in x else json.loads
            )
            parsed: Any = parser(x)
            assert isinstance(parsed, list)
            return len(parsed) + 1
        if isinstance(x, (list, tuple)):
            return len(x) + 1
        return 1

    df["selections_performed"] = df["prev_selections_performed"].apply(
        parse_and_count
    )

    cols_to_drop = ["prev_selections_performed"]
    if "selection_performed" in df.columns:
        cols_to_drop.append("selection_performed")
    df = df.drop(columns=cols_to_drop, errors="ignore")

    return df


def add_metadata_columns(
    df: pd.DataFrame,
    columns: dict[str, str | int | float | None],
) -> pd.DataFrame:
    """
    Add constant metadata columns to DataFrame.

    Replaces nu shell operations like:
        open {input} |
            insert afa_method {wildcards.method} |
            insert dataset {wildcards.dataset} |
            ...
            save {output}

    Args:
        df: Input DataFrame
        columns: Dict mapping column names to values

    Returns:
        DataFrame with new columns added
    """
    df = df.copy()
    for col_name, value in columns.items():
        df[col_name] = value
    return df


def pivot_long_classifier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot classifier predictions from wide to long (tidy) format.

    Replaces nu shell:
        open {input} |
            each { |row|
                [
                    ($row | insert classifier "builtin" | insert predicted_class $row.builtin_predicted_class),
                    ($row | insert classifier "external" | insert predicted_class $row.external_predicted_class)
                ]
            } |
            flatten |
            reject builtin_predicted_class external_predicted_class |
                save {output}

    Args:
        df: DataFrame with builtin_predicted_class and external_predicted_class columns

    Returns:
        DataFrame with classifier and predicted_class columns in long format
    """
    df = df.copy()

    classifier_cols = [c for c in df.columns if c.endswith("_predicted_class")]

    if not classifier_cols:
        df["classifier"] = "none"
        df["predicted_class"] = None
        return df

    available_classifier_cols = [
        col for col in classifier_cols if bool(df[col].notna().any())
    ]

    if not available_classifier_cols:
        df["classifier"] = "none"
        df["predicted_class"] = None
        return df

    id_vars = [c for c in df.columns if c not in classifier_cols]

    df_long = df.melt(
        id_vars=id_vars,
        value_vars=available_classifier_cols,
        var_name="classifier_type",
        value_name="predicted_class",
    )

    df_long["classifier"] = df_long["classifier_type"].str.replace(
        "_predicted_class", ""
    )
    df_long = df_long.drop(columns=["classifier_type"])

    return df_long


def validate_and_consolidate_budgets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that train and eval hard budgets match, consolidate soft budget params.

    Replaces nu shell validate_hard_budget_and_soft_budget_param rule:
        - Asserts train_hard_budget == eval_hard_budget for all rows
        - Renames train_hard_budget to hard_budget, drops eval_hard_budget
        - Asserts only one of train/eval soft_budget_param is set per row
        - Creates consolidated soft_budget_param column

    Args:
        df: DataFrame with budget columns

    Returns:
        DataFrame with consolidated budget columns

    Raises:
        ValueError: If validation fails
    """
    df = df.copy()

    if "train_hard_budget" in df.columns and "eval_hard_budget" in df.columns:
        train_filled = df["train_hard_budget"].fillna("_NaN_marker")
        eval_filled = df["eval_hard_budget"].fillna("_NaN_marker")
        mismatched = train_filled != eval_filled
        if mismatched.any():
            msg = (
                f"train_hard_budget and eval_hard_budget mismatch in "
                f"{len(mismatched)} rows"
            )
            raise ValueError(msg)

        df["hard_budget"] = df["train_hard_budget"]
        df = df.drop(columns=["train_hard_budget", "eval_hard_budget"])

    if (
        "train_soft_budget_param" in df.columns
        and "eval_soft_budget_param" in df.columns
    ):

        def is_empty(val: object) -> bool:
            if is_missing_value(val):
                return True
            return isinstance(val, str) and val in {"", NA_REP}

        both_set = df.apply(
            lambda row: not is_empty(row["train_soft_budget_param"])
            and not is_empty(row["eval_soft_budget_param"]),
            axis=1,
        )
        if bool(both_set.any()):
            msg = f"Both soft budget params set in {both_set.sum()} rows"
            raise ValueError(msg)

        def consolidate(row: pd.Series) -> str | float | None:
            train_value = row["train_soft_budget_param"]
            if not is_empty(train_value):
                assert isinstance(train_value, (str, float))
                return train_value
            eval_value = row["eval_soft_budget_param"]
            if not is_empty(eval_value):
                assert isinstance(eval_value, (str, float))
                return eval_value
            return ""

        df["soft_budget_param"] = df.apply(consolidate, axis=1)
        df = df.drop(
            columns=["train_soft_budget_param", "eval_soft_budget_param"]
        )

    return df


def transform_eval_data(
    input_path: Path,
    output_path: Path,
    afa_method: str,
    dataset: str | None = None,
    train_seed: int | None = None,
    eval_seed: int | None = None,
    hard_budget: int | None = None,
    soft_budget_param: float | None = None,
) -> None:
    """
    Full transformation pipeline for evaluation data.

    Takes the raw output from the eval_afa_method function and transforms it into
    the format expected by plotting scripts. This is the legacy interface.

    Args:
        input_path: Path to raw evaluation CSV file
        output_path: Path to save transformed CSV file
        afa_method: Name of the AFA method being evaluated
        dataset: Dataset name (optional)
        train_seed: Seed used during training (optional)
        eval_seed: Seed used during evaluation (optional)
        hard_budget: Hard budget used (optional)
        soft_budget_param: Soft budget parameter used (optional)
    """
    df = pd.read_csv(input_path)

    df = count_selections(df)

    metadata: dict[str, str | int | float | None] = {
        "afa_method": afa_method,
        "dataset": dataset,
        "train_seed": train_seed,
        "eval_seed": eval_seed,
        "hard_budget": hard_budget,
        "soft_budget_param": soft_budget_param,
    }
    df = add_metadata_columns(df, metadata)

    df = pivot_long_classifier(df)

    expected_columns = [
        "afa_method",
        "classifier",
        "dataset",
        "selections_performed",
        "predicted_class",
        "true_class",
        "train_seed",
        "eval_seed",
        "hard_budget",
        "soft_budget_param",
    ]

    final_columns = [col for col in expected_columns if col in df.columns]
    df = df[final_columns]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, na_rep=NA_REP)


def parse_col_arg(col_arg: str) -> tuple[str, str]:
    """Parse 'key=value' argument into (key, value) tuple."""
    if "=" not in col_arg:
        msg = f"Invalid column format: {col_arg}. Expected 'key=value'"
        raise ValueError(msg)
    key, value = col_arg.split("=", 1)
    return key.strip(), value.strip()


def cmd_count_selections(args: argparse.Namespace) -> None:
    """CLI handler for count_selections operation."""
    df = pd.read_csv(args.input_path)
    df = count_selections(df)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False, na_rep=NA_REP)


def cmd_add_metadata(args: argparse.Namespace) -> None:
    """CLI handler for add_metadata operation."""
    df = pd.read_csv(args.input_path)

    if not args.columns:
        msg = "add_metadata requires at least one --col argument"
        raise ValueError(msg)

    metadata: dict[str, str | int | float | None] = dict(
        parse_col_arg(c) for c in args.columns
    )
    df = add_metadata_columns(df, metadata)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False, na_rep=NA_REP)


def cmd_pivot_long_classifier(args: argparse.Namespace) -> None:
    """CLI handler for pivot_long_classifier operation."""
    df = pd.read_csv(args.input_path)
    df = pivot_long_classifier(df)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False, na_rep=NA_REP)


def cmd_validate_budgets(args: argparse.Namespace) -> None:
    """CLI handler for validate_budgets operation."""
    df = pd.read_csv(args.input_path)
    df = validate_and_consolidate_budgets(df)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False, na_rep=NA_REP)


def cmd_transform(args: argparse.Namespace) -> None:
    """CLI handler for full transform pipeline (legacy interface)."""
    transform_eval_data(
        input_path=args.input_path,
        output_path=args.output_path,
        afa_method=args.afa_method,
        dataset=args.dataset,
        train_seed=args.train_seed,
        eval_seed=args.eval_seed,
        hard_budget=args.hard_budget,
        soft_budget_param=args.soft_budget_param,
    )


def main() -> None:
    """Run the main entry point with subcommand routing."""
    parser = argparse.ArgumentParser(
        description="Transform evaluation data for plotting and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Count selections from JSON list
    python transform_eval_data.py count_selections input.csv output.csv

    # Add metadata columns
    python transform_eval_data.py add_metadata input.csv output.csv \\
        --col afa_method=aaco --col dataset=cube --col train_seed=42

    # Pivot classifiers to long format
    python transform_eval_data.py pivot_long_classifier input.csv output.csv

    # Validate and consolidate budget columns
    python transform_eval_data.py validate_budgets input.csv output.csv

    # Full transformation pipeline (legacy)
    python transform_eval_data.py transform input.csv output.csv my_method \\
        --dataset cube --train_seed 42
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands"
    )

    p_count = subparsers.add_parser(
        "count_selections",
        help="Convert prev_selections_performed to selections_performed count",
    )
    p_count.add_argument("input_path", type=Path, help="Input CSV path")
    p_count.add_argument("output_path", type=Path, help="Output CSV path")
    p_count.set_defaults(func=cmd_count_selections)

    p_meta = subparsers.add_parser(
        "add_metadata",
        help="Add constant metadata columns",
    )
    p_meta.add_argument("input_path", type=Path, help="Input CSV path")
    p_meta.add_argument("output_path", type=Path, help="Output CSV path")
    p_meta.add_argument(
        "--col",
        action="append",
        dest="columns",
        required=True,
        help="Column to add (format: key=value). Can be repeated.",
    )
    p_meta.set_defaults(func=cmd_add_metadata)

    p_pivot = subparsers.add_parser(
        "pivot_long_classifier",
        help="Pivot classifier predictions to long (tidy) format",
    )
    p_pivot.add_argument("input_path", type=Path, help="Input CSV path")
    p_pivot.add_argument("output_path", type=Path, help="Output CSV path")
    p_pivot.set_defaults(func=cmd_pivot_long_classifier)

    p_validate = subparsers.add_parser(
        "validate_budgets",
        help="Validate and consolidate budget columns",
    )
    p_validate.add_argument("input_path", type=Path, help="Input CSV path")
    p_validate.add_argument("output_path", type=Path, help="Output CSV path")
    p_validate.set_defaults(func=cmd_validate_budgets)

    p_transform = subparsers.add_parser(
        "transform",
        help="Full transformation pipeline (legacy interface)",
    )
    p_transform.add_argument("input_path", type=Path, help="Input CSV path")
    p_transform.add_argument("output_path", type=Path, help="Output CSV path")
    p_transform.add_argument(
        "afa_method", type=str, help="Name of the AFA method"
    )
    p_transform.add_argument(
        "--dataset", type=nullable_str, help="Dataset name (optional)"
    )
    p_transform.add_argument(
        "--train_seed", type=nullable_int, help="Training seed (optional)"
    )
    p_transform.add_argument(
        "--eval_seed", type=nullable_int, help="Evaluation seed (optional)"
    )
    p_transform.add_argument(
        "--hard_budget", type=nullable_int, help="Hard budget (optional)"
    )
    p_transform.add_argument(
        "--soft_budget_param",
        type=nullable_float,
        help="Soft budget parameter (optional)",
    )
    p_transform.set_defaults(func=cmd_transform)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
