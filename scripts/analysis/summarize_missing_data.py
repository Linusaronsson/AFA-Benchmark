"""Summarize the missing-training-data evaluation matrix."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import cast

import pandas as pd

_EXPERIMENT_PATTERN = re.compile(
    r"^method-(?P<method>.+?)\+mechanism-(?P<mechanism>.+?)"
    r"\+p-(?P<p>.+?)\+strategy-(?P<strategy>.+?)"
    r"\+instance-(?P<instance>\d+)$"
)
_RESTORATION_PATTERN = re.compile(
    r"mechanism-(?P<mechanism>[^/]+)\+p-(?P<p>[^/]+)/"
    r"instance-(?P<instance>\d+)/(?P<strategy>[^/]+)/"
    r"(?P<split>train|val)\.bundle/manifest\.json$"
)
_BASE_METHOD = {
    "aaco_doubly_robust": "aaco",
    "dime_feature_marginal_ipw": "dime",
}
_STRATEGY_DISPLAY = {
    "complete": "Complete data",
    "restricted": "Restricted",
    "mean_fill": "Mean completion",
    "zero_fill": "Zero fill (AACO only)",
    "pvae_label_conditioned": "PVAE (label-conditioned)",
    "pvae_label_free": "PVAE (label-free)",
    "pvae_oracle": "PVAE (oracle)",
    "true_completion": "True completion",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--instance-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--action-output", type=Path, required=True)
    parser.add_argument("--restoration-output", type=Path, required=True)
    return parser.parse_args()


def _metadata(path: Path) -> dict[str, object]:
    match = _EXPERIMENT_PATTERN.fullmatch(path.parent.name)
    if match is None:
        msg = f"Could not parse experiment metadata from {path}."
        raise ValueError(msg)
    values = match.groupdict()
    strategy = str(values["strategy"])
    return {
        "method": str(values["method"]),
        "mechanism": str(values["mechanism"]),
        "p": float(str(values["p"])),
        "strategy": strategy,
        "instance": int(str(values["instance"])),
        "strategy_display_name": _STRATEGY_DISPLAY[strategy],
    }


def _final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    final = frame.groupby("idx", sort=False, as_index=False).tail(1).copy()
    all_stopped = (final["action_performed"] == 0).to_numpy().all()
    if not bool(all_stopped):
        msg = "Every evaluated episode must end in a stop action."
        raise ValueError(msg)
    return final


def _instance_metrics(
    path: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    frame = pd.read_parquet(path)
    final = _final_rows(frame)
    metadata = _metadata(path)
    external_prediction = final["external_predicted_class"]
    has_missing_prediction = external_prediction.isna().to_numpy().any()
    if bool(has_missing_prediction):
        msg = f"External classifier predictions are missing from {path}."
        raise ValueError(msg)

    selections = final["prev_selections_performed"].map(len)
    result = metadata | {
        "n_samples": len(final),
        "accuracy": float((external_prediction == final["true_class"]).mean()),
        "mean_selections": float(selections.mean()),
        "mean_cost": float(final["accumulated_cost"].mean()),
        "forced_stop_rate": float(final["forced_stop"].mean()),
    }

    action_counts: dict[int, int] = {}
    for episode in final["prev_selections_performed"]:
        for selection in episode:
            selection_idx = int(selection)
            action_counts[selection_idx] = (
                action_counts.get(selection_idx, 0) + 1
            )
    action_rows = [
        metadata
        | {
            "selection": selection,
            "acquisitions_per_sample": count / len(final),
            "share_of_acquisitions": count
            / max(sum(action_counts.values()), 1),
        }
        for selection, count in sorted(action_counts.items())
    ]
    return result, action_rows


def _add_complete_data_gap(frame: pd.DataFrame) -> pd.DataFrame:
    references = frame.loc[
        frame["strategy"] == "complete",
        [
            "method",
            "instance",
            "accuracy",
        ],
    ].rename(
        columns={"method": "reference_method", "accuracy": "complete_accuracy"}
    )
    output = frame.copy()
    output["reference_method"] = output["method"].replace(_BASE_METHOD)
    output = output.merge(
        references,
        on=["reference_method", "instance"],
        how="left",
        validate="many_to_one",
    )
    output["accuracy_gap_to_complete"] = (
        output["accuracy"] - output["complete_accuracy"]
    )
    return output.drop(columns="reference_method")


def _aggregate(instance_frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "method",
        "mechanism",
        "p",
        "strategy",
        "strategy_display_name",
    ]
    metric_columns = [
        "accuracy",
        "accuracy_gap_to_complete",
        "mean_selections",
        "mean_cost",
        "forced_stop_rate",
    ]
    grouped = instance_frame.groupby(group_columns, dropna=False)
    mean = grouped[metric_columns].mean().add_suffix("_mean")
    sem = grouped[metric_columns].sem().add_suffix("_sem")
    count = cast("pd.Series", grouped["instance"].nunique()).to_frame(
        name="n_instances"
    )
    return count.join(mean).join(sem).reset_index()


def _restoration_metrics(input_root: Path) -> pd.DataFrame:
    missing_data_root = input_root.parents[2]
    rows: list[dict[str, object]] = []
    restored_root = missing_data_root / "views" / "restored" / input_root.name
    for path in sorted(restored_root.glob("**/manifest.json")):
        match = _RESTORATION_PATTERN.search(path.as_posix())
        if match is None:
            continue
        manifest = json.loads(path.read_text())
        metadata = manifest["metadata"]
        values = match.groupdict()
        rows.append(
            {
                "mechanism": values["mechanism"],
                "p": float(values["p"]),
                "instance": int(values["instance"]),
                "strategy": values["strategy"],
                "strategy_display_name": _STRATEGY_DISPLAY[values["strategy"]],
                "split": values["split"],
                "imputation_rmse": metadata.get("imputation_rmse"),
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    args = _parse_args()
    instance_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    for path in sorted(args.input_root.glob("*/eval_data.parquet")):
        instance_result, experiment_actions = _instance_metrics(path)
        instance_rows.append(instance_result)
        action_rows.extend(experiment_actions)
    if not instance_rows:
        msg = f"No evaluation parquet files found under {args.input_root}."
        raise FileNotFoundError(msg)

    instance_frame = _add_complete_data_gap(
        pd.DataFrame.from_records(instance_rows)
    )
    summary_frame = _aggregate(instance_frame)
    action_frame = pd.DataFrame.from_records(action_rows)
    restoration_frame = _restoration_metrics(args.input_root)

    for output in [
        args.instance_output,
        args.summary_output,
        args.action_output,
        args.restoration_output,
    ]:
        output.parent.mkdir(parents=True, exist_ok=True)
    instance_frame.to_csv(args.instance_output, index=False)
    summary_frame.to_csv(args.summary_output, index=False)
    action_frame.to_csv(args.action_output, index=False)
    restoration_frame.to_csv(args.restoration_output, index=False)


if __name__ == "__main__":
    main()
