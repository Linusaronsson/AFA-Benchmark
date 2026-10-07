"""
Transform a raw evaluation table into a plotting-ready table.

The raw table identifies itself (ADR 0002): identity comes from its columns,
and its provenance record is copied into the output's schema metadata. The
transform only derives `n_selections_performed`, melts the prediction
columns into `classifier` and `predicted_class`, and normalises nullable
dtypes. Identity arguments are checks: one that disagrees with a non-null
column raises, and a missing or null column is filled from its argument.
"""

import argparse
import ast
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sized

import pandas as pd

from afabench.evaluation.provenance import (
    evaluation_table_provenance,
    save_evaluation_table,
)
from afabench.evaluation.schemas import (
    IDENTITY_DTYPES,
    PreProvenanceSavedEvaluationSchema,
    SavedEvaluationSchema,
)

# Command-line argument checked against each identity column
IDENTITY_ARGUMENTS = {
    "afa_method": "method",
    "dataset": "dataset",
    "initializer": "initializer",
    "train_seed": "train_seed",
    "train_hard_budget": "train_hard_budget",
    "train_soft_budget_param": "train_soft_budget_param",
    "eval_soft_budget_param": "eval_soft_budget_param",
}


class IdentityArgumentMismatchError(ValueError):
    """Raised when an identity argument disagrees with the table's column."""


def parse_nullable(s: str) -> str | None:
    if s == "null":
        return None
    return s


def count_selections(selections: object) -> int:
    if isinstance(selections, str):
        return len(ast.literal_eval(selections))
    return len(cast("Sized", selections))


def check_or_fill_identity(
    df: pd.DataFrame, column: str, argument: str | None
) -> None:
    """
    Check `column` against `argument`, or fill it if missing or null.

    An omitted argument (`None`) checks nothing; the string `null` is an
    explicit null value.
    """
    dtype = IDENTITY_DTYPES[column]
    if argument is None:
        if column not in df:
            df[column] = pd.Series(pd.NA, index=df.index, dtype=dtype)
        return
    expected = pd.Series(
        parse_nullable(argument), index=df.index, dtype=object
    ).astype(dtype)
    if column not in df or bool(df[column].isna().all()):
        df[column] = expected
        return
    # Non-null identity is constant per table, so one value disagrees or none
    disagreeing = df[column].ne(expected).fillna(value=True)
    if bool(disagreeing.any()):
        value = df.loc[disagreeing, column].iloc[0]
        msg = (
            f"Argument --{IDENTITY_ARGUMENTS[column]}={argument} "
            f"disagrees with column {column!r}, which holds {value!r}."
        )
        raise IdentityArgumentMismatchError(msg)


def main() -> None:
    """Transform evaluation data in a single pipeline."""
    parser = argparse.ArgumentParser(
        description="Transform evaluation data in a single pipeline"
    )
    parser.add_argument("--input_path", type=Path, help="Input parquet path")
    parser.add_argument("--output_path", type=Path, help="Output parquet path")
    parser.add_argument(
        "--method", type=str, help="AFA method name, checked if given."
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset key, checked if given."
    )
    parser.add_argument(
        "--initializer",
        type=str,
        help="Initializer name used for this run, checked if given.",
    )
    parser.add_argument(
        "--train_seed",
        type=str,
        help="Training seed, checked if given. `null` if not applicable.",
    )
    parser.add_argument(
        "--train_hard_budget",
        type=str,
        help="Training hard budget, checked if given. `null` if not applicable.",
    )
    parser.add_argument(
        "--train_soft_budget_param",
        type=str,
        help=(
            "Training soft budget parameter, checked if given. "
            "`null` if not applicable."
        ),
    )
    parser.add_argument(
        "--eval_soft_budget_param",
        type=str,
        help=(
            "Evaluation soft budget parameter, checked if given. "
            "`null` if not applicable."
        ),
    )

    args = parser.parse_args()

    df = pd.read_parquet(args.input_path)
    if "episode_id" in df or "step" in df:
        # Validate before dtype conversion so malformed events are not
        # coerced. Tables saved before ADR 0002 keep their own column set.
        if set(df.columns) == set(
            PreProvenanceSavedEvaluationSchema.to_schema().columns
        ):
            PreProvenanceSavedEvaluationSchema.validate(df)
        else:
            SavedEvaluationSchema.validate(df)
        df["n_selections_performed"] = df["step"].astype("UInt64")
    else:
        # Legacy analyses can use stored lengths even for partial logs. Do not
        # infer episode identity or implicitly migrate these artifacts.
        df["n_selections_performed"] = (
            df["prev_selections_performed"]
            .map(count_selections)
            .astype("UInt64")
        )
        # Legacy logs predate per-episode instance identity (ADR 0004)
        df["generation_index"] = pd.NA
        df["split_index"] = pd.NA

    for column, argument in IDENTITY_ARGUMENTS.items():
        check_or_fill_identity(df, column, getattr(args, argument))
    # No argument names these; a table saved before ADR 0002 leaves them null
    for column in ["dataset_realization_index", "eval_split"]:
        check_or_fill_identity(df, column, None)

    df = df.astype(
        {
            "action_performed": "UInt64",
            "builtin_predicted_class": "UInt64",
            "external_predicted_class": "UInt64",
            "true_class": "UInt64",
            "accumulated_cost": "Float64",
            "forced_stop": "boolean",
            "generation_index": "UInt64",
            "split_index": "UInt64",
            **IDENTITY_DTYPES,
        }
    )

    # Pivot long on classifier type
    df = df.rename(
        columns={
            "builtin_predicted_class": "builtin",
            "external_predicted_class": "external",
        }
    ).melt(
        # Index is everything else except stuff we don't care about for plotting
        id_vars=[
            "generation_index",
            "split_index",
            "action_performed",
            "true_class",
            "accumulated_cost",
            "forced_stop",
            "n_selections_performed",
            *IDENTITY_DTYPES,
        ],
        value_vars=["builtin", "external"],
        var_name="classifier",
        value_name="predicted_class",
    )

    save_evaluation_table(
        df,
        args.output_path,
        provenance=evaluation_table_provenance(args.input_path),
    )


if __name__ == "__main__":
    main()
