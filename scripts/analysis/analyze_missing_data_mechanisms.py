"""Mechanism checks for grouped CUBE-NM and restored training data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from afabench.core.bundle_system.bundle import load_bundle
from scripts.analysis.route_redundancy import primary_metric
from scripts.analysis.summarize_missing_data import _final_rows, _metadata

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from afabench.core.types import AFADataset

_PAIR_KEYS = [
    "dataset",
    "method",
    "mechanism",
    "p",
    "instance",
    "eval_hard_budget",
]


def cube_nm_episode_metrics(
    contexts: NDArray[np.int64],
    sequences: list[list[int]],
    *,
    block_size: int = 10,
) -> dict[str, float | int]:
    """Measure whether a route reads context, then follows its branch."""
    if len(contexts) != len(sequences):
        msg = "contexts and sequences must have equal lengths."
        raise ValueError(msg)
    context_first = np.asarray(
        [bool(sequence) and sequence[0] == 0 for sequence in sequences]
    )
    eligible = np.asarray(
        [
            is_first and len(sequence) > 1
            for is_first, sequence in zip(
                context_first,
                sequences,
                strict=True,
            )
        ]
    )
    correct_next = np.asarray(
        [
            is_eligible and (sequence[1] - 1) // block_size == int(context)
            for context, sequence, is_eligible in zip(
                contexts,
                sequences,
                eligible,
                strict=True,
            )
        ]
    )
    allocation: list[float] = []
    for context, sequence, is_first in zip(
        contexts,
        sequences,
        context_first,
        strict=True,
    ):
        if not is_first or len(sequence) <= 1:
            continue
        blocks = [(selection - 1) // block_size for selection in sequence[1:]]
        allocation.append(float(np.mean(np.asarray(blocks) == int(context))))
    return {
        "n_episodes": len(sequences),
        "context_first_rate": float(context_first.mean()),
        "n_correct_next_eligible": int(eligible.sum()),
        "correct_next_rate": (
            float(correct_next[eligible].mean())
            if bool(eligible.any())
            else float("nan")
        ),
        "correct_block_allocation": (
            float(np.mean(allocation)) if allocation else float("nan")
        ),
    }


def _evaluation_paths(eval_root: Path) -> list[Path]:
    return sorted(eval_root.glob("dataset-*/**/eval_data.parquet"))


def _positioned_final_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover each episode's position despite batch-local trace indices."""
    history_lengths = frame["prev_selections_performed"].map(len)
    batch_starts = history_lengths.eq(0) & frame["idx"].eq(0)
    if frame.empty or not bool(batch_starts.iloc[0]):
        msg = "Evaluation trace does not start with a batch-local index zero."
        raise ValueError(msg)

    annotated = frame.copy()
    annotated["_batch"] = batch_starts.cumsum() - 1
    batch_sizes = (
        annotated.loc[history_lengths.eq(0)]
        .groupby("_batch", sort=True)["idx"]
        .count()
    )
    offsets = batch_sizes.cumsum().shift(fill_value=0).astype(int)
    final = _final_rows(annotated)
    final["_position"] = final["_batch"].map(offsets).astype(int) + final[
        "idx"
    ].astype(int)
    final = final.sort_values("_position")
    if final["_position"].tolist() != list(range(len(final))):
        msg = "Could not recover one final row per evaluation position."
        raise ValueError(msg)
    return final


def _sampled_indices(
    n_dataset: int,
    n_samples: int,
    seed: int,
) -> NDArray[np.int64]:
    generator = torch.Generator().manual_seed(seed)
    subset = torch.randperm(n_dataset, generator=generator)[:n_samples]
    order = torch.randperm(len(subset), generator=generator)
    return subset[order].numpy()


def _evaluation_indices(
    dataset_labels: NDArray[np.int64],
    observed_labels: NDArray[np.int64],
    seed: int,
) -> NDArray[np.int64]:
    """Infer sequential versus seeded subset evaluation from saved labels."""
    n_samples = len(observed_labels)
    candidates = {
        "sequential": np.arange(n_samples),
        "sampled": _sampled_indices(
            len(dataset_labels),
            n_samples,
            seed,
        ),
    }
    matching = {
        name: indices
        for name, indices in candidates.items()
        if np.array_equal(dataset_labels[indices], observed_labels)
    }
    distinct = {
        tuple(indices.tolist()): indices for indices in matching.values()
    }
    if len(distinct) != 1:
        msg = (
            "Could not uniquely align evaluation episodes with dataset "
            f"instances; matching orders: {sorted(matching)}."
        )
        raise ValueError(msg)
    return next(iter(distinct.values()))


def path_fidelity(
    eval_root: Path,
    dataset_root: Path,
    *,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contexts_by_instance: dict[int, NDArray[np.int64]] = {}
    labels_by_instance: dict[int, NDArray[np.int64]] = {}
    for path in _evaluation_paths(eval_root):
        metadata = _metadata(path)
        if metadata["dataset"] != "cube_nm":
            continue
        instance = int(cast("Any", metadata["instance"]))
        if instance not in contexts_by_instance:
            bundle_path = (
                dataset_root / "cube_nm" / str(instance) / f"{split}.bundle"
            )
            dataset = cast(
                "AFADataset",
                cast("object", load_bundle(bundle_path)[0]),
            )
            features, labels = dataset.get_all_data()
            contexts_by_instance[instance] = (
                features[:, :5]
                .argmax(dim=1)
                .cpu()
                .numpy()
                .astype(np.int64, copy=False)
            )
            labels_by_instance[instance] = (
                labels.argmax(dim=1).cpu().numpy()
                if labels.ndim > 1
                else labels.cpu().numpy()
            ).astype(np.int64, copy=False)

        frame = pd.read_parquet(path)
        final = _positioned_final_rows(frame)
        eval_seeds = frame["eval_seed"].unique()
        if len(eval_seeds) != 1:
            msg = f"Expected one evaluation seed in {path}."
            raise ValueError(msg)
        indices = _evaluation_indices(
            labels_by_instance[instance],
            final["true_class"].astype(int).to_numpy(),
            int(cast("Any", eval_seeds[0])),
        )
        contexts = contexts_by_instance[instance][indices]
        sequences = [
            [int(value) for value in sequence]
            for sequence in final["prev_selections_performed"]
        ]
        rows.append(
            metadata
            | cube_nm_episode_metrics(
                contexts,
                sequences,
            )
        )
    return pd.DataFrame.from_records(rows)


def _primary_scores(instance_metrics: pd.DataFrame) -> pd.Series:
    return cast(
        "pd.Series",
        instance_metrics.apply(
            lambda row: row[primary_metric(str(row["dataset"]))],
            axis=1,
        ),
    )


def stepwise_effects(instance_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = instance_metrics.loc[
        instance_metrics["strategy"].isin(
            ["restricted", "pvae_label_conditioned", "pvae_stepwise"]
        )
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    frame["primary_score"] = _primary_scores(frame)
    if frame.duplicated([*_PAIR_KEYS, "strategy"]).any():
        msg = "Duplicate strategy cells in instance metrics."
        raise ValueError(msg)
    pivot = frame.pivot_table(
        index=_PAIR_KEYS,
        columns="strategy",
        values="primary_score",
        aggfunc="first",
    )
    required = {
        "restricted",
        "pvae_label_conditioned",
        "pvae_stepwise",
    }
    if not required.issubset(pivot.columns):
        return pd.DataFrame()
    output = pivot.dropna(subset=list(required)).reset_index()
    output["episode_start_gain"] = (
        output["pvae_label_conditioned"] - output["restricted"]
    )
    output["stepwise_gain"] = output["pvae_stepwise"] - output["restricted"]
    output["stepwise_minus_episode_start"] = (
        output["pvae_stepwise"] - output["pvae_label_conditioned"]
    )
    return output


def generator_quality_pairs(
    instance_metrics: pd.DataFrame,
    restoration: pd.DataFrame,
) -> pd.DataFrame:
    rmse = restoration.loc[
        (restoration["split"] == "val")
        & restoration["strategy"].isin(
            ["pvae_label_conditioned", "pvae_oracle"]
        )
    ].copy()
    if rmse.empty:
        return pd.DataFrame()
    rmse_keys = ["dataset", "mechanism", "p", "instance"]
    if rmse.duplicated([*rmse_keys, "strategy"]).any():
        msg = "Duplicate restoration-RMSE cells."
        raise ValueError(msg)
    rmse_pivot = rmse.pivot_table(
        index=rmse_keys,
        columns="strategy",
        values="imputation_rmse",
        aggfunc="first",
    )
    required = {"pvae_label_conditioned", "pvae_oracle"}
    if not required.issubset(rmse_pivot.columns):
        return pd.DataFrame()
    rmse_pivot = rmse_pivot.dropna(subset=list(required)).reset_index()
    rmse_pivot["rmse_improvement"] = (
        rmse_pivot["pvae_label_conditioned"] - rmse_pivot["pvae_oracle"]
    )

    scores = instance_metrics.loc[
        instance_metrics["strategy"].isin(sorted(required))
    ].copy()
    scores["primary_score"] = _primary_scores(scores)
    if scores.duplicated([*_PAIR_KEYS, "strategy"]).any():
        msg = "Duplicate generator-score cells."
        raise ValueError(msg)
    score_pivot = scores.pivot_table(
        index=_PAIR_KEYS,
        columns="strategy",
        values="primary_score",
        aggfunc="first",
    )
    if not required.issubset(score_pivot.columns):
        return pd.DataFrame()
    score_pivot = score_pivot.dropna(subset=list(required)).reset_index()
    score_pivot["score_improvement"] = (
        score_pivot["pvae_oracle"] - score_pivot["pvae_label_conditioned"]
    )
    return score_pivot.merge(
        rmse_pivot[rmse_keys + ["rmse_improvement"]],
        on=rmse_keys,
        how="inner",
        validate="many_to_one",
    )


def _score_final_rows(final: pd.DataFrame, metric: str) -> float:
    truth = final["true_class"]
    prediction = final["external_predicted_class"]
    if metric == "accuracy":
        return float((truth == prediction).mean())
    return float(
        f1_score(
            truth,
            prediction,
            average="macro",
            zero_division=cast("Any", 0),
        )
    )


def _trace_columns(final: pd.DataFrame) -> pd.DataFrame:
    output = cast(
        "pd.DataFrame",
        final.sort_values("_position")[
            [
                "_position",
                "prev_selections_performed",
                "external_predicted_class",
                "true_class",
                "accumulated_cost",
                "forced_stop",
            ]
        ].copy(),
    )
    history = cast("pd.Series", output["prev_selections_performed"])
    output["prev_selections_performed"] = history.map(tuple)
    return output.reset_index(drop=True)


def dime_invariance(eval_root: Path) -> pd.DataFrame:
    paths: dict[tuple[str, int, float, str, str, float], Path] = {}
    for path in _evaluation_paths(eval_root):
        metadata = _metadata(path)
        if metadata["method"] != "dime":
            continue
        key = (
            str(metadata["dataset"]),
            int(cast("Any", metadata["instance"])),
            float(cast("Any", metadata["eval_hard_budget"])),
            str(metadata["strategy"]),
            str(metadata["mechanism"]),
            float(cast("Any", metadata["p"])),
        )
        paths[key] = path

    rows: list[dict[str, Any]] = []
    for key, path in paths.items():
        dataset, instance, budget, strategy, mechanism, probability = key
        if strategy != "true_completion":
            continue
        complete_key = (
            dataset,
            instance,
            budget,
            "complete",
            "none",
            0.0,
        )
        complete_path = paths.get(complete_key)
        if complete_path is None:
            continue
        complete = _positioned_final_rows(pd.read_parquet(complete_path))
        restored = _positioned_final_rows(pd.read_parquet(path))
        metric = primary_metric(dataset)
        rows.append(
            {
                "dataset": dataset,
                "instance": instance,
                "eval_hard_budget": budget,
                "mechanism": mechanism,
                "p": probability,
                "exact_trace_match": _trace_columns(complete).equals(
                    _trace_columns(restored)
                ),
                "primary_score_delta": (
                    _score_final_rows(restored, metric)
                    - _score_final_rows(complete, metric)
                ),
                "complete_path": str(complete_path),
                "true_completion_path": str(path),
            }
        )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("extra/output/missing_data"),
    )
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-dime-invariance", action="store_true")
    args = parser.parse_args()

    eval_root = args.root / "eval" / args.split / args.namespace
    summary_root = args.root / "summary" / args.split / args.namespace
    output = args.output_dir or args.root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    instance_metrics = pd.read_csv(summary_root / "instance_metrics.csv")
    restoration = pd.read_csv(summary_root / "restoration_rmse.csv")
    outputs = {
        "path_fidelity": path_fidelity(
            eval_root,
            args.root / "datasets" / args.namespace,
            split=args.split,
        ),
        "stepwise_effects": stepwise_effects(instance_metrics),
        "generator_quality": generator_quality_pairs(
            instance_metrics,
            restoration,
        ),
        "dime_invariance": dime_invariance(eval_root),
    }
    for name, frame in outputs.items():
        path = output / f"{name}_{args.namespace}_{args.split}.csv"
        frame.to_csv(path, index=False)
        print(f"wrote {path} ({len(frame)} rows)")

    invariance = outputs["dime_invariance"]
    exact_matches = cast("pd.Series", invariance["exact_trace_match"])
    if args.require_dime_invariance and (
        invariance.empty or not bool(exact_matches.astype(bool).all())
    ):
        msg = "DIME true-completion invariance failed."
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
