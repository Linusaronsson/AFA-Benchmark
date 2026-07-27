"""Build the exact processed-feature acquisition schema for CoAI NHANES."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

GROUP_NAMES = [
    "BUN",
    "Age",
    "Alkaline Phosphatase",
    "CBC w/Diff",
    "Calcium",
    "Cholesterol",
    "Creatinine",
    "Height",
    "Hemoglobin",
    "Physical Activity",
    "CBC Auto",
    "Potassium",
    "Pulse Pressure",
    "Red Blood Cells",
    "Sedimentation Rate",
    "Serum Albumin",
    "Serum Protein",
    "Sex",
    "Sodium",
    "Systolic BP",
    "Total Bilirubin",
    "Uric Acid",
    "Urine Albumin",
    "Urine Glucose",
    "Urinalysis",
    "Weight",
    "SGOT",
]
LOW_COST_FLOOR = 0.001


def _read_groups(path: Path) -> dict[str, int]:
    groups: dict[str, int] = {}
    for line in path.read_text().splitlines():
        feature, group = line.split(",")
        groups[feature.strip().lower()] = int(group)
    return groups


def _read_costs(path: Path) -> dict[str, float]:
    raw: dict[str, float] = {}
    for line in path.read_text().splitlines():
        feature, value = line.split("\t")
        raw[feature.strip().lower()] = (
            float(value) if value != "?" else float("nan")
        )
    known_mean = float(np.nanmean(list(raw.values())))
    return {
        feature: known_mean if np.isnan(cost) else cost
        for feature, cost in raw.items()
    }


def build_schema(source_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_dir / "X_nhanes_binary.csv")
    features = frame.drop(columns="Unnamed: 0").columns.tolist()
    groups = _read_groups(source_dir / "feature_groups.txt")
    costs = _read_costs(source_dir / "feature_costs.txt")

    rows: list[dict[str, int | float | str]] = []
    for feature_index, feature_name in enumerate(features):
        matches = [
            source_feature
            for source_feature in groups
            if source_feature in feature_name.lower()
        ]
        if len(matches) != 1:
            msg = (
                f"Expected one source mapping for {feature_name}, got "
                f"{matches}."
            )
            raise ValueError(msg)
        source_feature = matches[0]
        group_id = groups[source_feature]
        rows.append(
            {
                "feature_index": feature_index,
                "feature_name": feature_name,
                "source_feature": source_feature,
                "group_id": group_id,
                "group_name": GROUP_NAMES[group_id],
                "source_cost": costs[source_feature],
            }
        )

    schema = pd.DataFrame(rows)
    if sorted(schema["group_id"].unique().tolist()) != list(
        range(len(GROUP_NAMES))
    ):
        msg = "NHANES group identifiers must cover 0 through 26."
        raise ValueError(msg)

    group_source_costs = {
        int(cast("Any", group_id)): int(n_costs)
        for group_id, n_costs in schema.groupby("group_id")["source_cost"]
        .nunique()
        .items()
    }
    bad = [
        group_id
        for group_id, n_costs in group_source_costs.items()
        if n_costs != 1
    ]
    if bad:
        msg = f"Source costs disagree inside groups: {bad}"
        raise ValueError(msg)
    group_sizes = schema.groupby("group_id")["feature_index"].transform(
        "count"
    )
    schema["group_cost"] = schema["source_cost"] + LOW_COST_FLOOR
    schema["feature_cost"] = schema["group_cost"] / group_sizes
    return schema


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("extra/data/nhanes_mortality/source"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("extra/data/nhanes_mortality/schema.csv"),
    )
    args = parser.parse_args()

    schema = build_schema(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    schema.to_csv(args.output, index=False, float_format="%.12g")
    print(
        f"Wrote {len(schema)} features in "
        f"{schema['group_id'].nunique()} groups to {args.output}"
    )


if __name__ == "__main__":
    main()
