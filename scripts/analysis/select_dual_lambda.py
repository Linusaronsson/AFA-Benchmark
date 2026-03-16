from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd

DEFAULT_GROUP_COLUMNS = (
    "method",
    "train_initializer",
    "eval_initializer",
    "eval_soft_budget_param",
    "eval_hard_budget",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select dual_lambda values from CUBE-NM-AR aggregate summaries "
            "under a risky unsafe-stop budget."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help=(
            "Root directory containing dual_lambda-*/aggregate_summary.csv "
            "subdirectories."
        ),
    )
    parser.add_argument(
        "--risk-budget",
        type=float,
        required=True,
        help="Upper bound on risky_unsafe_stop_rate_mean.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional CSV path for selected lambdas.",
    )
    parser.add_argument(
        "--group-columns",
        nargs="+",
        default=list(DEFAULT_GROUP_COLUMNS),
        help="Columns defining one tuning problem.",
    )
    return parser.parse_args()


def _load_summaries(input_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    pattern = re.compile(r"dual_lambda-([^/]+)")

    for path in sorted(input_root.glob("dual_lambda-*/aggregate_summary.csv")):
        match = pattern.search(str(path.parent))
        if match is None:
            continue
        dual_lambda = float(match.group(1))
        summary_df = pd.read_csv(path)
        summary_df["dual_lambda"] = dual_lambda
        rows.append(summary_df)

    if not rows:
        msg = f"No aggregate_summary.csv files found under {input_root}."
        raise FileNotFoundError(msg)

    return pd.concat(rows, ignore_index=True)


def _choose_lambda(
    df: pd.DataFrame,
    *,
    risk_budget: float,
    group_columns: list[str],
) -> pd.DataFrame:
    missing_columns = [
        column for column in group_columns if column not in df.columns
    ]
    if missing_columns:
        msg = f"Missing grouping columns: {missing_columns}"
        raise ValueError(msg)

    feasible = df[df["risky_unsafe_stop_rate_mean"] <= risk_budget].copy()
    if feasible.empty:
        msg = (
            "No dual_lambda entries satisfy the requested risk budget. "
            f"Budget={risk_budget}."
        )
        raise ValueError(msg)

    sort_columns = [
        *group_columns,
        "mean_cost_mean",
        "accuracy_mean",
        "dual_lambda",
    ]
    sort_ascending: Any = [True] * len(group_columns) + [True, False, True]
    feasible = cast("Any", feasible).sort_values(
        by=sort_columns,
        ascending=sort_ascending,
    )
    best = feasible.groupby(group_columns, as_index=False).head(1).copy()
    best = best.sort_values(group_columns).reset_index(drop=True)
    return best


def main() -> None:
    args = parse_args()
    summary = _load_summaries(args.input_root)
    selected = _choose_lambda(
        summary,
        risk_budget=args.risk_budget,
        group_columns=args.group_columns,
    )

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(args.output_path, index=False)

    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
