import argparse
import ast
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sized

import pandas as pd

from afabench.evaluation.schemas import SavedEvaluationSchema


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

    df = pd.read_parquet(args.input_path)
    if "episode_id" in df or "step" in df:
        # Validate before dtype conversion so malformed events are not coerced.
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

    df = df.astype(
        {
            "action_performed": "UInt64",
            "builtin_predicted_class": "UInt64",
            "external_predicted_class": "UInt64",
            "true_class": "UInt64",
            "accumulated_cost": "Float64",
            "forced_stop": "boolean",
            "eval_seed": "UInt64",
            "eval_hard_budget": "Float64",
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
            "action_performed",
            "true_class",
            "accumulated_cost",
            "forced_stop",
            "eval_seed",
            "eval_hard_budget",
            "n_selections_performed",
        ],
        value_vars=["builtin", "external"],
        var_name="classifier",
        value_name="predicted_class",
    )

    # Add some columns provided as args
    metadata = {
        "afa_method": (args.method, "string"),
        "dataset": (args.dataset, "string"),
        "initializer": (args.initializer, "string"),
        "train_seed": (parse_nullable(args.train_seed), "UInt64"),
        "train_hard_budget": (
            parse_nullable(args.train_hard_budget),
            "Float64",
        ),
        "train_soft_budget_param": (
            parse_nullable(args.train_soft_budget_param),
            "Float64",
        ),
        "eval_soft_budget_param": (
            parse_nullable(args.eval_soft_budget_param),
            "Float64",
        ),
    }
    for name, (value, dtype) in metadata.items():
        df[name] = pd.Series(value, index=df.index, dtype=object).astype(dtype)

    df.to_parquet(args.output_path, index=False)


if __name__ == "__main__":
    main()
