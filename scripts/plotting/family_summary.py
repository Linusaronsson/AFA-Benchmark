"""Average paired state representations before resampling dataset instances."""

from typing import Any, cast

import numpy as np
import pandas as pd

from afabench.plotting.methods import METHOD_FAMILIES, PRIMARY_METHODS

FAMILY_MEMBERS = {
    family: tuple(
        method
        for method in PRIMARY_METHODS
        if METHOD_FAMILIES[method] == family
    )
    for family in ("dime", "gdfs", "aaco", "jafa", "ol", "odin")
}
FAMILY_LABELS = {
    "dime": "DIME",
    "gdfs": "GDFS",
    "aaco": "AACO",
    "jafa": "JAFA",
    "ol": "OL",
    "odin": "ODIN",
}
ARM_COLUMNS = {
    "complete": "ceiling_abs",
    "restricted": "direct_abs",
    "pvae_label_conditioned": "generative_abs",
}


def average_states(frame: pd.DataFrame) -> pd.DataFrame:
    """Require matched complete, restricted and restored scores for each state."""
    identity = [
        "dataset",
        "instance",
        "train_hard_budget",
        "eval_hard_budget",
    ]
    keys = [*identity, "method", "mechanism", "p", "strategy"]
    selected = frame.loc[
        frame["strategy"].isin(ARM_COLUMNS)
        & frame["method"].isin(
            [m for members in FAMILY_MEMBERS.values() for m in members]
        )
    ].copy()
    if selected.duplicated(keys).any():
        message = "duplicate family-summary scientific cells"
        raise ValueError(message)
    complete = selected.loc[selected["strategy"] == "complete"]
    if complete.duplicated([*identity, "method"]).any():
        message = "ambiguous complete-data reference"
        raise ValueError(message)
    reference = complete[[*identity, "method", "score"]].rename(
        columns={"score": "ceiling_abs"}
    )
    missing = selected.loc[selected["strategy"] != "complete"]
    paired = missing.merge(
        reference,
        on=[*identity, "method"],
        how="left",
        validate="many_to_one",
    )
    if paired[["score", "ceiling_abs"]].isna().any().any():
        message = "missing family score or complete-data reference"
        raise ValueError(message)
    output = []
    cell_keys = [
        "dataset",
        "instance",
        "eval_hard_budget",
        "mechanism",
        "p",
        "metric",
    ]
    grouped = cast("Any", paired.groupby(cell_keys, sort=True))
    for key, cell in grouped:
        for family, members in FAMILY_MEMBERS.items():
            family_cell = cell.loc[cell["method"].isin(members)]
            if len(family_cell) != 2 * len(members):
                msg = f"incomplete paired family {family} at {key}"
                raise ValueError(msg)
            budget = family_cell["train_hard_budget"].iloc[0]
            if not family_cell["train_hard_budget"].eq(budget).all():
                message = f"unpaired family training budgets at {key}"
                raise ValueError(message)
            wide = family_cell.pivot_table(
                index="method",
                columns="strategy",
                values="score",
                aggfunc="first",
            ).reindex(members)
            if wide.isna().any().any() or wide.shape != (len(members), 2):
                msg = f"unpaired family treatments at {key}"
                raise ValueError(msg)
            record = dict(zip(cell_keys, key, strict=True))
            record.update(
                method=family,
                train_hard_budget=budget,
                n_variants=len(members),
                ceiling_abs=float(family_cell["ceiling_abs"].mean()),
                direct_abs=float(wide["restricted"].mean()),
                generative_abs=float(wide["pvae_label_conditioned"].mean()),
            )
            output.append(record)
    result = pd.DataFrame(output)
    if result.empty:
        message = "no family-summary instances"
        raise ValueError(message)
    return result


def summarize_families(instances: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap the five paired instance means with fixed state weights."""
    draws = np.random.default_rng(0).integers(0, 5, (2000, 5))
    records = []
    keys = ["dataset", "method", "mechanism", "p", "metric"]
    grouped = cast("Any", instances.groupby(keys, sort=True))
    for key, raw_group in grouped:
        group = raw_group.sort_values("instance")
        if (
            len(group) != 5
            or group["instance"].nunique() != 5
            or not group["train_hard_budget"]
            .eq(group["train_hard_budget"].iloc[0])
            .all()
            or not group["eval_hard_budget"]
            .eq(group["eval_hard_budget"].iloc[0])
            .all()
        ):
            msg = f"expected five matched family instances at {key}"
            raise ValueError(msg)
        record = dict(zip(keys, key, strict=True))
        record.update(
            n_instances=5, n_variants=int(group["n_variants"].iloc[0])
        )
        for column in ARM_COLUMNS.values():
            values = group[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                msg = f"non-finite family scores at {key}"
                raise ValueError(msg)
            record[column] = float(values.mean())
            low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
            record[f"{column}_lo"] = float(low)
            record[f"{column}_hi"] = float(high)
        record["damage"] = record["ceiling_abs"] - record["direct_abs"]
        record["gain"] = record["generative_abs"] - record["direct_abs"]
        records.append(record)
    return pd.DataFrame(records)
