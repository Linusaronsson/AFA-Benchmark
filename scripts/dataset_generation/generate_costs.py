"""
Generate per-feature acquisition costs for tabular datasets.

Writes:
  extra/data/misc/feature_costs/{dataset_key}.csv

Modes:
- uniform: uniform random costs in [low, high]
- cube: half cheap / half costly
    - deterministic (default): cheap half all == --cheap, costly half all == --costly
    - random (--random): cheap half ~ U(cheap_low, cheap_high), costly half ~ U(costly_low, costly_high)
      (optionally shuffled)

All costs are normalized to mean == 1.0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import torch

DEFAULT_OUTDIR: Final[Path] = Path("extra/data/misc/feature_costs")

ERR_NUM_FEATURES_POSITIVE: Final[str] = "--num-features must be > 0"
ERR_CUBE_RANDOM_ORDERED_RANGES: Final[str] = (
    "cube --random expects non-overlapping or ordered ranges: "
    "require costly_low > cheap_high to guarantee 'costly' > 'cheap'."
)


def normalize_mean_1(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    m = x.mean()
    if not torch.isfinite(m) or m.abs().item() < eps:
        msg = f"Cannot normalize costs: mean={m.item()}"
        raise ValueError(msg)
    return x / m


def write_csv(dataset_key: str, costs: torch.Tensor, outdir: Path) -> Path:
    if costs.ndim != 1:
        msg = f"Costs must be 1D (F,), got {tuple(costs.shape)}"
        raise ValueError(msg)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{dataset_key}.csv"
    path.write_text(",".join(f"{v:.6f}" for v in costs.tolist()) + "\n")
    return path


def _check_range(name: str, lo: float, hi: float) -> None:
    if lo <= 0 or hi <= 0:
        msg = f"{name} range must be positive (got low={lo}, high={hi})"
        raise ValueError(msg)
    if hi <= lo:
        msg = f"{name} requires high > low (got low={lo}, high={hi})"
        raise ValueError(msg)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate feature-cost CSVs (mean-normalized to 1.0)."
    )
    p.add_argument("--dataset-key", required=True)
    p.add_argument("-F", "--num-features", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)

    sub = p.add_subparsers(dest="mode", required=True)

    pu = sub.add_parser("uniform")
    pu.add_argument("--low", type=float, default=1.0)
    pu.add_argument("--high", type=float, default=10.0)

    pc = sub.add_parser("cube")
    pc.add_argument(
        "--shuffle", action="store_true", help="Shuffle feature assignment"
    )
    pc.add_argument(
        "--random",
        action="store_true",
        help="Sample costs from separate cheap/costly ranges (still split 50/50)",
    )

    # deterministic cube (default)
    pc.add_argument(
        "--cheap", type=float, default=1.0, help="Deterministic cheap cost"
    )
    pc.add_argument(
        "--costly", type=float, default=10.0, help="Deterministic costly cost"
    )

    # random cube (only used when --random)
    pc.add_argument("--cheap-low", type=float, default=1.0)
    pc.add_argument("--cheap-high", type=float, default=2.0)
    pc.add_argument("--costly-low", type=float, default=5.0)
    pc.add_argument("--costly-high", type=float, default=10.0)

    return p


def _gen_uniform(
    num_features: int, low: float, high: float, g: torch.Generator
) -> torch.Tensor:
    _check_range("uniform", low, high)
    return (high - low) * torch.rand(num_features, generator=g) + low


def _gen_cube(
    num_features: int, a: argparse.Namespace, g: torch.Generator
) -> torch.Tensor:
    n_cheap = num_features // 2
    n_costly = num_features - n_cheap  # +1 costly when num_features is odd

    if a.random:
        _check_range("cheap", a.cheap_low, a.cheap_high)
        _check_range("costly", a.costly_low, a.costly_high)
        if a.costly_low <= a.cheap_high:
            msg = ERR_CUBE_RANDOM_ORDERED_RANGES
            raise ValueError(msg)

        cheap = (a.cheap_high - a.cheap_low) * torch.rand(
            n_cheap, generator=g
        ) + a.cheap_low
        costly = (a.costly_high - a.costly_low) * torch.rand(
            n_costly, generator=g
        ) + a.costly_low
        costs = torch.cat([cheap, costly], dim=0)
    else:
        if a.cheap <= 0 or a.costly <= 0:
            msg = f"cube requires positive costs (cheap={a.cheap}, costly={a.costly})"
            raise ValueError(msg)
        if a.costly <= a.cheap:
            msg = f"cube expects costly > cheap (cheap={a.cheap}, costly={a.costly})"
            raise ValueError(msg)
        costs = torch.tensor([a.cheap] * n_cheap + [a.costly] * n_costly)

    if a.shuffle:
        costs = costs[torch.randperm(num_features, generator=g)]
    return costs


def main() -> None:
    p = _build_parser()
    a = p.parse_args()

    num_features = a.num_features
    if num_features <= 0:
        msg = ERR_NUM_FEATURES_POSITIVE
        raise ValueError(msg)

    g = torch.Generator().manual_seed(a.seed)

    if a.mode == "uniform":
        costs = _gen_uniform(num_features, a.low, a.high, g)
    else:  # cube
        costs = _gen_cube(num_features, a, g)

    costs = normalize_mean_1(costs.to(torch.float32))
    outpath = write_csv(a.dataset_key, costs, a.outdir)

    print(f"Wrote: {outpath}")
    print(
        f"dataset_key={a.dataset_key}  F={num_features}  mode={a.mode}  seed={a.seed}"
    )
    print(
        f"min={costs.min().item():.6f}  max={costs.max().item():.6f}  mean={costs.mean().item():.6f}"
    )


if __name__ == "__main__":
    main()
