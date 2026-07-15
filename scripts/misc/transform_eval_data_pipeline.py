import argparse
import ast
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sized

import polars as pl


def parse_nullable(s: str) -> str | None:
    if s == "null":
        return None
    return s


def count_selections(selections: object) -> int:
    if isinstance(selections, str):
        return len(ast.literal_eval(selections))
    return len(cast("Sized", selections))


def main() -> None:
    """Transform evaluation data in a single pipeline."""
    parser = argparse.ArgumentParser(
        description="Transform evaluation data in a single pipeline"
    )
    parser.add_argument("--input_path", type=Path, help="Input parquet path")
    parser.add_argument("--output_path", type=Path, help="Output parquet path")
    parser.add_argument("--method", type=str, help="AFA method name")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument(
        "--initializer",
        type=str,
        help="Initializer name used for this run.",
    )
    parser.add_argument(
        "--train_seed",
        type=str,
        help="Training seed. `null` if not applicable.",
    )
    parser.add_argument(
        "--train_hard_budget",
        type=str,
        help="Training hard budget. `null` if not applicable.",
    )
    parser.add_argument(
        "--train_soft_budget_param",
        type=str,
        help="Training soft budget parameter. `null` if not applicable.",
    )
    parser.add_argument(
        "--eval_soft_budget_param",
        type=str,
        help="Evaluation soft budget parameter. `null` if not applicable.",
    )

    args = parser.parse_args()

    df = pl.read_parquet(args.input_path).with_columns(
        action_performed=pl.col("action_performed").cast(
            pl.UInt64, strict=False
        ),
        builtin_predicted_class=pl.col("builtin_predicted_class").cast(
            pl.UInt64, strict=False
        ),
        external_predicted_class=pl.col("external_predicted_class").cast(
            pl.UInt64, strict=False
        ),
        true_class=pl.col("true_class").cast(pl.UInt64, strict=False),
        accumulated_cost=pl.col("accumulated_cost").cast(
            pl.Float64, strict=False
        ),
        idx=pl.col("idx").cast(pl.UInt64, strict=False),
        forced_stop=pl.col("forced_stop").cast(pl.Boolean, strict=False),
        eval_seed=pl.col("eval_seed").cast(pl.UInt64, strict=False),
        eval_hard_budget=pl.col("eval_hard_budget").cast(
            pl.Float64, strict=False
        ),
    )

    # Change prev_selections_performed (a history of selections) to instead just be the number of selections performed, which is the same as the time step
    df = df.with_columns(
        n_selections_performed=pl.col(
            "prev_selections_performed"
        ).map_elements(count_selections, return_dtype=pl.UInt64)
    ).drop("prev_selections_performed")

    # Pivot long on classifier type
    df = df.rename(
        {
            "builtin_predicted_class": "builtin",
            "external_predicted_class": "external",
        }
    ).unpivot(
        on=["builtin", "external"],
        # Index is everything else except stuff we don't care about for plotting
        index=[
            "action_performed",
            "true_class",
            "accumulated_cost",
            "forced_stop",
            "eval_seed",
            "eval_hard_budget",
            "n_selections_performed",
        ],
        variable_name="classifier",
        value_name="predicted_class",
    )

    # Add some columns provided as args
    df = df.with_columns(
        afa_method=pl.lit(args.method, dtype=pl.String),
        dataset=pl.lit(args.dataset, dtype=pl.String),
        initializer=pl.lit(args.initializer, dtype=pl.String),
        train_seed=pl.lit(parse_nullable(args.train_seed), dtype=pl.UInt64),
        train_hard_budget=pl.lit(
            parse_nullable(args.train_hard_budget), dtype=pl.Float64
        ),
        train_soft_budget_param=pl.lit(
            parse_nullable(args.train_soft_budget_param), dtype=pl.Float64
        ),
        eval_soft_budget_param=pl.lit(
            parse_nullable(args.eval_soft_budget_param), dtype=pl.Float64
        ),
    )

    df.write_parquet(args.output_path)


if __name__ == "__main__":
    main()
