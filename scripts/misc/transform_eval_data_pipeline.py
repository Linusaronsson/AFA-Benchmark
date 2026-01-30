import argparse
from pathlib import Path

import polars as pl


def parse_nullable(s: str) -> str | None:
    if s == "null":
        return None
    return s


def main() -> None:
    """Transform evaluation data in a single pipeline."""
    parser = argparse.ArgumentParser(
        description="Transform evaluation data in a single pipeline"
    )
    parser.add_argument("--input_path", type=Path, help="Input CSV path")
    parser.add_argument("--output_path", type=Path, help="Output CSV path")
    parser.add_argument("--method", type=str, help="AFA method name")
    parser.add_argument("--dataset", type=str, help="Dataset name")
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

    df = pl.read_csv(
        args.input_path,
        schema={
            "prev_selections_performed": pl.String,
            "action_performed": pl.Int64,
            "builtin_predicted_class": pl.Int64,
            "external_predicted_class": pl.Int64,
            "true_class": pl.Int64,
            "accumulated_cost": pl.Float64,
            "idx": pl.Int64,
            "forced_stop": pl.Boolean,
            "eval_seed": pl.Int64,
            "eval_hard_budget": pl.Float64,
        },
        null_values=["null"],
    )

    # Change prev_selections_performed (a history of selections) to instead just be the number of selections performed, which is the same as the time step
    df = df.with_columns(
        n_selections_performed=pl.col("prev_selections_performed").len()
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
        train_seed=pl.lit(parse_nullable(args.train_seed), dtype=pl.Int64),
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

    df.write_csv(args.output_path)


if __name__ == "__main__":
    main()
