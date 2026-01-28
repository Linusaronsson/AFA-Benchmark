"""
Chain all evaluation data transformations in a single Python script.

This replaces the multi-step nushell approach with pure Python.
"""
# ADD THIS
# # Either train_soft_budget_param is set, or eval_soft_budget_param is set, but not both. This means that one of them must always be null
# assert (
#     df["train_soft_budget_param"].is_null()
#     | df["eval_soft_budget_param"].is_null()
# ).all(), (
#     "Both train_soft_budget_param and eval_soft_budget_param cannot be set. Choose one."
# )
# df = df.with_columns(
#     soft_budget_param=pl.coalesce(
#         "train_soft_budget_param", "eval_soft_budget_param"
#     )
# )

import argparse
import ast
import json
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


def norm_na(x: object) -> object:
    if is_missing_value(x):
        return None
    if isinstance(x, str) and x.strip().lower() == NA_REP:
        return None
    return x


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
        # train_filled = df["train_hard_budget"].fillna("_NaN_marker")
        # eval_filled = df["eval_hard_budget"].fillna("_NaN_marker")
        # mismatched = train_filled != eval_filled
        # Only check the hard budget evaluation
        train_norm = df["train_hard_budget"].map(norm_na)
        eval_norm = df["eval_hard_budget"].map(norm_na)
        mismatched = (eval_norm.notna()) & (train_norm != eval_norm)

        if mismatched.any():
            msg = (
                f"train_hard_budget and eval_hard_budget mismatch in "
                f"{mismatched.sum()} rows"
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


def transform_eval_data_pipeline(
    input_path: Path,
    output_path: Path,
    method: str,
    dataset: str,
    train_seed: int,
    train_hard_budget: str,
    train_soft_budget_param: str,
    eval_soft_budget_param: str,
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
        "eval_soft_budget_param": float(eval_soft_budget_param)
        if eval_soft_budget_param != "null"
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
    parser.add_argument(
        "eval_soft_budget_param", help="Evaluation soft budget parameter"
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
        eval_soft_budget_param=args.eval_soft_budget_param,
    )


if __name__ == "__main__":
    main()
