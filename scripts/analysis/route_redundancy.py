"""
Route structure VS. what the trained methods achieve.

Per dataset, use the fixed shared classifier purely as a scoring rule to map the
space of size-b feature subsets, then place what the trained policies actually
achieve (from summary.csv) against that reference.

    a_rand      mean accuracy of a random size-b subset
    a_best      best fixed size-b subset            (best static route)
    route_corr  correlation of per-instance correctness among the top-quantile
                subsets (high = interchangeable/flat, low = complementary/planning)
    selection_sensitivity a_best - a_rand           (is there a good static route)
    planning_gain         acc_complete - a_best     (adaptive policy beats the
                best static route = non-myopic advantage; uses the trained policy
                because a per-instance-best over random subsets just saturates)

The gap that matters is restoration vs the direct approach, not vs mean fill
(mean fill is a lazy generative baseline, restricted is the direct domain):
    restoration_gain    acc[pvae_label_conditioned] - acc[restricted]
    missingness_damage  acc[complete] - acc[restricted]
Reads per-strategy achieved accuracy from summary.csv and per-feature acquisition
divergence from action_rates.csv, so route structure can be tied to whether the
direct policy finds the good routes and the generative model restores them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml

from afabench.core.bundle_system.bundle import load_bundle

if TYPE_CHECKING:
    from afabench.core.types import AFADataset


def score_subsets(
    clf: Any,  # noqa: ANN401
    feats: torch.Tensor,
    true: torch.Tensor,
    feature_shape: torch.Size,
    b: int,
    k: int,
    gen: np.random.Generator,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64]]:
    n, f = feats.shape
    correct = np.empty((k, n), dtype=bool)
    for j in range(k):
        s = torch.as_tensor(
            gen.choice(f, size=b, replace=False), device=feats.device
        )
        fm = torch.zeros((n, f), dtype=torch.bool, device=feats.device)
        fm[:, s] = True
        pred = clf(feats * fm, fm, feature_shape=feature_shape).argmax(-1)
        correct[j] = (pred == true).cpu().numpy()
    return correct, correct.mean(1)


def route_metrics(
    correct: npt.NDArray[np.bool_],
    accs: npt.NDArray[np.float64],
    top_frac: float,
) -> dict[str, float]:
    # correlation among the top-quantile subsets (always defined, unlike an
    # absolute near-max threshold which is near-empty for planning datasets)
    m = max(5, int(np.ceil(top_frac * len(accs))))
    top = correct[np.argsort(accs)[::-1][:m]].astype(float)
    top = top[top.var(1) > 0]  # constant rows have undefined correlation
    if len(top) >= 2:
        cm = np.corrcoef(top)
        route_corr = float(np.nanmean(cm[np.triu_indices_from(cm, 1)]))
    else:
        route_corr = float("nan")
    a_rand = float(accs.mean())
    a_best = float(accs.max())
    return {
        "a_rand": a_rand,
        "a_best": a_best,
        "route_corr": route_corr,
        "selection_sensitivity": a_best - a_rand,
    }


def achieved(summary: pd.DataFrame, dataset: str) -> dict[str, float | str]:
    """Per-strategy achieved accuracy (mean over methods) and restoration gap."""
    df = summary.loc[summary["dataset"] == dataset]
    out: dict[str, float | str] = {}
    complete = df.loc[df["strategy"] == "complete", "accuracy_mean"]
    if len(complete):
        out["acc_complete"] = float(complete.mean())
    miss = df.loc[df["mechanism"] != "none"]
    if not len(miss):
        return out
    p = float(miss["p"].max())
    cell = miss.loc[miss["p"] == p]
    acc = cell.groupby("strategy")["accuracy_mean"].mean().to_dict()
    out["gap_p"] = p
    out["gap_mechanism"] = ",".join(
        sorted(cell["mechanism"].astype(str).unique())
    )
    for strat in ("restricted", "mean_fill", "pvae_label_conditioned"):
        if strat in acc:
            out[f"acc_{strat}"] = float(acc[strat])
    # the gap that matters is restoration vs the direct approach, not vs mean fill
    if {"mean_fill", "pvae_label_conditioned"} <= acc.keys():
        out["restoration_gap"] = (
            acc["pvae_label_conditioned"] - acc["mean_fill"]
        )
    if {"restricted", "pvae_label_conditioned"} <= acc.keys():
        out["restoration_gain"] = (
            acc["pvae_label_conditioned"] - acc["restricted"]
        )
    return out


def usage_divergence(
    action_rates: pd.DataFrame, dataset: str
) -> dict[str, float]:
    """Per-feature acquisition-profile L1 distance from complete, over methods."""
    df = action_rates.loc[action_rates["dataset"] == dataset]
    if not len(df):
        return {}
    prof = df.pivot_table(
        index=["method", "strategy"],
        columns="selection",
        values="acquisitions_per_sample",
        aggfunc="mean",
        fill_value=0.0,
    )
    out: dict[str, float] = {}
    for strat in ("restricted", "pvae_label_conditioned"):
        dists = []
        for method in prof.index.get_level_values("method").unique():
            if (method, "complete") in prof.index and (
                method,
                strat,
            ) in prof.index:
                ref = prof.loc[(method, "complete")]
                dists.append(
                    float(np.abs(prof.loc[(method, strat)] - ref).sum()) / 2
                )
        if dists:
            out[f"usage_l1_{strat}"] = float(np.mean(dists))
    return out


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def process_dataset(
    clf_path: Path,
    a: argparse.Namespace,
    budgets: dict[str, list[int]],
    gen: np.random.Generator,
    summary: pd.DataFrame | None,
    action_rates: pd.DataFrame | None,
) -> dict[str, Any] | None:
    dataset = clf_path.name[len("dataset-") : -len(".bundle")]
    if a.datasets and dataset not in a.datasets:
        return None
    ds_path = (
        a.root / "datasets" / a.namespace / dataset / "0" / f"{a.split}.bundle"
    )
    if not ds_path.exists():
        print(f"skip {dataset}: missing {ds_path}")
        return None
    clf, _ = load_bundle(clf_path, device=torch.device(a.device))
    # load_bundle deserializes to the concrete dataset behind the Loadeable type
    ds = cast("AFADataset", load_bundle(ds_path)[0])  # pyright: ignore[reportInvalidCast]
    feats, labels = ds.get_all_data()
    if len(feats) > a.max_samples:
        keep = torch.as_tensor(
            gen.choice(len(feats), size=a.max_samples, replace=False)
        )
        feats, labels = feats[keep], labels[keep]
    feats = feats.to(a.device)
    true = labels.argmax(-1).to(a.device)
    b = min(max(budgets.get(dataset, budgets["default"])), feats.shape[1])
    correct, accs = score_subsets(
        clf, feats, true, ds.feature_shape, b, a.k, gen
    )
    row: dict[str, Any] = {
        "namespace": a.namespace,
        "dataset": dataset,
        "n_features": int(feats.shape[1]),
        "n_samples": len(feats),
        "b": b,
        "k": a.k,
    }
    row |= route_metrics(correct, accs, a.top_frac)
    if summary is not None:
        row |= achieved(summary, dataset)
    if action_rates is not None:
        row |= usage_divergence(action_rates, dataset)
    if "acc_complete" in row:
        row["planning_gain"] = row["acc_complete"] - row["a_best"]
        if "acc_restricted" in row:
            # adaptive advantage the direct approach loses to missingness
            row["missingness_damage"] = (
                row["acc_complete"] - row["acc_restricted"]
            )
    print(
        f"{dataset:14s} b={b:2d} a_best={row['a_best']:.3f} "
        f"route_corr={row['route_corr']:.3f} "
        f"plan_gain={row.get('planning_gain', float('nan')):.3f} "
        f"restore_gain={row.get('restoration_gain', float('nan')):.3f} "
        f"damage={row.get('missingness_damage', float('nan')):.3f}"
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=Path, default=Path("extra/output/missing_data")
    )
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument(
        "--budgets",
        type=Path,
        default=Path("extra/workflow/conf/eval_hard_budgets/all.yaml"),
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--k", type=int, default=500)
    ap.add_argument("--top-frac", type=float, default=0.1)
    ap.add_argument("--max-samples", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    budgets = yaml.safe_load(a.budgets.read_text())["eval_hard_budgets"]
    gen = np.random.default_rng(a.seed)
    summary = _read(a.root / "summary" / a.split / a.namespace / "summary.csv")
    action_rates = _read(
        a.root / "summary" / a.split / a.namespace / "action_rates.csv"
    )

    clf_paths = sorted(
        (a.root / "classifier" / a.namespace).glob("dataset-*.bundle")
    )
    rows = [
        row
        for clf_path in clf_paths
        if (
            row := process_dataset(
                clf_path, a, budgets, gen, summary, action_rates
            )
        )
        is not None
    ]
    if not rows:
        msg = f"no classifiers under {a.root}/classifier/{a.namespace}"
        raise SystemExit(msg)
    out = (
        a.output or a.root / "analysis" / f"route_redundancy_{a.namespace}.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
