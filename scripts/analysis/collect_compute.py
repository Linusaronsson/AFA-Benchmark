"""Attribute cluster CPU time to direct learning and generative restoration."""

# The workflow never wrote a `benchmark:` for the training rules, so training
# cost is not in the artifact tree. It is recoverable anyway. The Snakemake
# controller log names every job's rule and wildcards next to its Slurm job id,
# and Slurm accounting still holds the CPU time, so joining the two labels every
# job with the cell it belongs to. Collect the two inputs on the cluster by
# grepping the controller log for its "submitted with SLURM jobid" lines, and by
# running sacct with JobID, Elapsed, TotalCPU, AllocCPUS and State.

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import cast

import pandas as pd

JOB_LINE = re.compile(
    r"Job (?P<snakemake_job>\d+) has been submitted with SLURM jobid (?P<slurm_job>\d+) "
    r"\(log: .*?/slurm_logs/rule_(?P<rule>[a-z_]+)/(?P<wildcards>[^/]*)/"
)

# Longest alternatives first, since several tokens are prefixes of others.
DATASET = r"cube_nm|cube"
METHOD = r"aaco_doubly_robust|aaco|dime_feature_marginal_ipw|dime|ol_without_mask|random_dummy|odin_model_free"
MECHANISM = r"mnar_logistic|mnar_self|mcar|mar|none"
STRATEGY = (
    r"pvae_label_conditioned|pvae_label_free|pvae_stepwise|pvae_oracle|"
    r"true_completion|mean_fill|zero_fill|restricted|complete"
)

# Each rule spells its wildcards in its own order.
WILDCARD_PATTERNS = {
    "train_missing_data_method_with_pretraining": re.compile(
        rf"^(?P<dataset>{DATASET})_(?P<method>{METHOD})_(?P<mechanism>{MECHANISM})_"
        rf"(?P<p>[\d.]+)_(?P<strategy>{STRATEGY})_(?P<instance>\d+)_(?P<train_budget>\d+)$"
    ),
    "train_missing_data_method_without_pretraining": re.compile(
        rf"^(?P<dataset>{DATASET})_(?P<method>{METHOD})_(?P<mechanism>{MECHANISM})_"
        rf"(?P<p>[\d.]+)_(?P<strategy>{STRATEGY})_(?P<instance>\d+)_(?P<train_budget>\d+)$"
    ),
    "eval_missing_data_method": re.compile(
        rf"^(?P<dataset>{DATASET})_(?P<method>{METHOD})_(?P<mechanism>{MECHANISM})_"
        rf"(?P<p>[\d.]+)_(?P<strategy>{STRATEGY})_(?P<instance>\d+)_"
        rf"(?P<train_budget>\d+)_(?P<eval_budget>\d+)$"
    ),
    "pretrain_missing_data_method": re.compile(
        rf"^(?P<method>{METHOD})_(?P<dataset>{DATASET})_(?P<mechanism>{MECHANISM})_"
        rf"(?P<p>[\d.]+)_(?P<strategy>{STRATEGY})_(?P<instance>\d+)$"
    ),
    "materialize_missing_training_view": re.compile(
        rf"^(?P<dataset>{DATASET})_(?P<mechanism>{MECHANISM})_(?P<p>[\d.]+)_"
        rf"(?P<strategy>{STRATEGY})_(?P<instance>\d+)$"
    ),
    "restore_missing_training_view": re.compile(
        rf"^(?P<dataset>{DATASET})_(?P<mechanism>{MECHANISM})_(?P<p>[\d.]+)_"
        rf"(?P<strategy>{STRATEGY})_(?P<instance>\d+)$"
    ),
    "pretrain_incomplete_restoration_pvae": re.compile(
        rf"^(?P<dataset>{DATASET})_(?P<mechanism>{MECHANISM})_(?P<p>[\d.]+)_(?P<instance>\d+)$"
    ),
    "pretrain_oracle_restoration_pvae": re.compile(
        rf"^(?P<dataset>{DATASET})_(?P<instance>\d+)$"
    ),
    "train_missing_data_shared_classifier": re.compile(
        rf"^(?P<dataset>{DATASET})_(?P<instance>\d+)$"
    ),
}


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    """Typed column access, since a bare lookup widens to include ndarray."""
    return cast("pd.Series", frame[name])


def parse_duration(value: str) -> float:
    """Seconds from a Slurm duration, which may carry days and fractional seconds."""
    value = value.strip()
    if not value or value in {"", "INVALID"}:
        return float("nan")
    days = 0
    if "-" in value:
        day_part, value = value.split("-", 1)
        days = int(day_part)
    parts = value.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = "0", parts[0], parts[1]
    else:
        hours, minutes, seconds = "0", "0", parts[0]
    return (
        days * 86400 + int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    )


def parse_joblog(path: Path) -> pd.DataFrame:
    rows = []
    unparsed = 0
    for line in path.read_text().splitlines():
        match = JOB_LINE.search(line)
        if match is None:
            continue
        rule = match["rule"]
        record = {
            "rule": rule,
            "slurm_job": int(match["slurm_job"]),
            "wildcards": match["wildcards"],
        }
        pattern = WILDCARD_PATTERNS.get(rule)
        if pattern is not None:
            fields = pattern.match(match["wildcards"])
            if fields is None:
                unparsed += 1
            else:
                record.update(fields.groupdict())
        rows.append(record)
    frame = pd.DataFrame(rows)
    if unparsed:
        print(
            f"warning: {unparsed} wildcard strings did not match their rule pattern"
        )
    return frame


def parse_sacct(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="|")
    # TotalCPU is populated on the batch step, not on the allocation row.
    job_id = _column(frame, "JobID").astype(str)
    batch = cast("pd.DataFrame", frame[job_id.str.endswith(".batch")]).copy()
    batch["slurm_job"] = (
        _column(batch, "JobID").astype(str).str.split(".").str[0].astype(int)
    )
    batch["cpu_seconds"] = (
        _column(batch, "TotalCPU").astype(str).map(parse_duration)
    )
    batch["wall_seconds"] = (
        _column(batch, "Elapsed").astype(str).map(parse_duration)
    )
    keep = ["slurm_job", "cpu_seconds", "wall_seconds", "AllocCPUS", "State"]
    return cast("pd.DataFrame", batch[keep])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joblog", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    jobs = parse_joblog(arguments.joblog)
    accounting = parse_sacct(arguments.sacct)
    joined = jobs.merge(accounting, on="slurm_job", how="left")

    missing = int(_column(joined, "cpu_seconds").isna().sum())
    if missing:
        print(
            f"warning: {missing} of {len(joined)} jobs had no accounting record"
        )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(arguments.output, index=False)

    print(f"jobs: {len(joined)}")
    grouped = joined.groupby("rule")["cpu_seconds"].agg(
        ["size", "sum", "median"]
    )
    summary = cast("pd.DataFrame", grouped).sort_values(
        by="sum", ascending=False
    )
    summary["sum_hours"] = summary["sum"] / 3600.0
    print(summary[["size", "sum_hours", "median"]].round(2).to_string())
    total_hours = float(_column(joined, "cpu_seconds").sum()) / 3600.0
    print(f"total CPU hours: {total_hours:.1f}")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
