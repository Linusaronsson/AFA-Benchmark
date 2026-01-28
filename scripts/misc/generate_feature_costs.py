from __future__ import annotations

import argparse
import zlib
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf

from afabench.common.registry import get_class

if TYPE_CHECKING:
    from afabench.common.custom_types import AFADataset

type CostArray = NDArray[np.floating]


def _generate_costs(
    rng: np.random.Generator,
    n_features: int,
    distribution: str,
    lognormal_mean: float,
    lognormal_sigma: float,
    uniform_low: float,
    uniform_high: float,
) -> CostArray:
    if distribution == "lognormal":
        costs = rng.lognormal(
            mean=lognormal_mean, sigma=lognormal_sigma, size=n_features
        )
    else:
        costs = rng.uniform(
            low=uniform_low, high=uniform_high, size=n_features
        )
    return costs / costs.mean()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic per-feature acquisition costs."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset config name (e.g. diabetes, physionet, ckd).",
    )
    parser.add_argument(
        "--config-dir",
        default="extra/conf/scripts/dataset_generation/dataset",
        help="Directory with dataset generation YAML configs.",
    )
    parser.add_argument(
        "--dataset-seed",
        type=int,
        default=None,
        help="Seed for datasets that require one (default: 0).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Output CSV path. Defaults to "
            "extra/data/misc/feature_costs/{dataset}.csv."
        ),
    )
    parser.add_argument(
        "--distribution",
        choices=["lognormal", "uniform"],
        default="lognormal",
        help="Distribution used to sample costs before rescaling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Defaults to crc32(dataset).",
    )
    parser.add_argument(
        "--lognormal-mean",
        type=float,
        default=0.0,
        help="Log-normal mean parameter.",
    )
    parser.add_argument(
        "--lognormal-sigma",
        type=float,
        default=1.0,
        help="Log-normal sigma parameter.",
    )
    parser.add_argument(
        "--uniform-low",
        type=float,
        default=0.5,
        help="Uniform distribution lower bound.",
    )
    parser.add_argument(
        "--uniform-high",
        type=float,
        default=1.5,
        help="Uniform distribution upper bound.",
    )
    return parser.parse_args()


def _load_dataset_from_config(
    dataset_key: str, config_dir: Path, dataset_seed: int | None
) -> AFADataset:
    config_path = config_dir / f"{dataset_key}.yaml"
    assert config_path.exists(), f"Missing config: {config_path}"
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        msg = f"Expected dict config at {config_path}, got {type(cfg)!r}."
        raise TypeError(msg)
    class_name = cfg.get("class_name")
    kwargs = dict(cfg.get("kwargs") or {})
    assert class_name, f"Missing class_name in {config_path}"
    dataset_cls = cast("type[AFADataset]", get_class(class_name))
    if dataset_cls.accepts_seed() and "seed" not in kwargs:
        kwargs["seed"] = 0 if dataset_seed is None else dataset_seed
    return dataset_cls(**kwargs)


def main() -> None:
    args = _parse_args()
    config_dir = Path(args.config_dir)
    dataset = _load_dataset_from_config(
        args.dataset, config_dir, args.dataset_seed
    )
    n_features = prod(dataset.feature_shape)
    seed = (
        args.seed
        if args.seed is not None
        else zlib.crc32(args.dataset.encode("utf-8")) & 0xFFFFFFFF
    )
    rng = np.random.default_rng(seed)
    costs = _generate_costs(
        rng=rng,
        n_features=n_features,
        distribution=args.distribution,
        lognormal_mean=args.lognormal_mean,
        lognormal_sigma=args.lognormal_sigma,
        uniform_low=args.uniform_low,
        uniform_high=args.uniform_high,
    )
    output_path = (
        Path(args.output)
        if args.output is not None
        else Path("extra/data/misc/feature_costs") / f"{args.dataset}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, costs, delimiter=",", fmt="%.6f")
    print(
        f"Wrote {n_features} costs to {output_path} (seed={seed}, "
        f"mean={costs.mean():.3f})."
    )


if __name__ == "__main__":
    main()
