"""Audit factual-native evaluation traces before scientific aggregation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import pandas as pd

from scripts.analysis.summarize_missing_data import _metadata

_REQUIRED_COLUMNS = {
    "action_performed",
    "source_idx",
    "selection_was_legal",
    "respected_native_availability",
    "feature_availability_fraction",
    "selection_availability_fraction",
}
_PROHIBITED_STRATEGIES = {"pvae_oracle", "true_completion"}


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return cast(
        "pd.Series",
        pd.to_numeric(cast("pd.Series", frame[column]), errors="raise"),
    )


def _constant_fraction(frame: pd.DataFrame, column: str, path: Path) -> float:
    values = _numeric(frame, column).drop_duplicates()
    if len(values) != 1:
        msg = f"{column} is not constant in {path}."
        raise ValueError(msg)
    value = float(values.iloc[0])
    if not 0.0 <= value <= 1.0:
        msg = f"{column} is outside [0, 1] in {path}."
        raise ValueError(msg)
    return value


def _audit_trace(path: Path, metadata: dict[str, object]) -> dict[str, object]:
    if metadata["strategy"] in _PROHIBITED_STRATEGIES:
        msg = f"Native evaluation uses prohibited strategy in {path}."
        raise ValueError(msg)

    frame = pd.read_parquet(path)
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        msg = f"Missing native audit columns in {path}: {', '.join(missing)}"
        raise ValueError(msg)
    if frame.empty:
        msg = f"Native evaluation is empty: {path}."
        raise ValueError(msg)

    legal = cast("pd.Series", frame["selection_was_legal"])
    respected = cast("pd.Series", frame["respected_native_availability"])
    illegal_actions = int((~legal.astype(bool)).sum())
    if legal.isna().any() or illegal_actions:
        msg = f"Found {illegal_actions} illegal native actions in {path}."
        raise ValueError(msg)
    if respected.isna().any() or not bool(respected.astype(bool).all()):
        msg = f"Native availability was not enforced in {path}."
        raise ValueError(msg)

    feature_fraction = _constant_fraction(
        frame, "feature_availability_fraction", path
    )
    selection_fraction = _constant_fraction(
        frame, "selection_availability_fraction", path
    )
    if feature_fraction == 1.0:
        msg = f"Native evaluation contains no missing source values: {path}."
        raise ValueError(msg)

    source_idx = _numeric(frame, "source_idx")
    if source_idx.isna().any() or bool((source_idx < 0).any()):
        msg = f"Invalid source indices in {path}."
        raise ValueError(msg)
    actions = _numeric(frame, "action_performed")
    return metadata | {
        "n_trace_rows": len(frame),
        "n_source_samples": int(source_idx.nunique()),
        "n_acquisitions": int((actions > 0).sum()),
        "n_illegal_actions": illegal_actions,
        "feature_availability_fraction": feature_fraction,
        "selection_availability_fraction": selection_fraction,
        "source_file": str(path),
    }


def audit_native_evaluations(input_root: Path) -> pd.DataFrame:
    """Return one verified row per factual-native evaluation artifact."""
    rows: list[dict[str, object]] = []
    for path in sorted(input_root.glob("dataset-*/**/eval_data.parquet")):
        metadata = _metadata(path)
        if metadata["mechanism"] == "native":
            rows.append(_audit_trace(path, metadata))

    if not rows:
        msg = f"No factual-native evaluations found under {input_root}."
        raise FileNotFoundError(msg)
    return pd.DataFrame.from_records(rows).sort_values(
        ["dataset", "method", "strategy", "instance"],
        kind="stable",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    report = audit_native_evaluations(arguments.input_root)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(arguments.output, index=False)
    print(f"verified native evaluations: {len(report)}")
    print(f"illegal actions: {int(report['n_illegal_actions'].sum())}")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
