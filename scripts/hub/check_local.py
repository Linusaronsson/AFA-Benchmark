"""Check that all bundles expected by the pipeline are present locally."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(
    0, str(Path(__file__).parents[2] / "extra" / "workflow" / "src")
)
from config import load_config  # pyright: ignore[reportMissingImports]

_DEFAULT_BUNDLE_TYPES = (
    "trained_methods",
    "pretrained_models",
    "trained_classifiers",
)


def _load_and_merge_configs(configfiles: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in configfiles:
        with path.open() as f:
            data = yaml.safe_load(f) or {}
        _deep_merge(merged, data)
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _expected_classifier_paths(cfg: dict[str, Any], tag: str) -> list[str]:
    paths = [
        f"extra/output/trained_classifiers/{tag}/dataset-{dataset}.bundle"
        for dataset in cfg["DATASETS"]
    ]
    paths += [
        f"extra/output/trained_classifiers/{tag}/method-{method}+dataset-{dataset}.bundle"
        for method in cfg["METHODS"]
        if method in cfg["METHOD_CLASSIFIER_SCRIPT_NAMES"]
        for dataset in cfg["DATASETS"]
    ]
    return paths


def _expected_pretrained_model_paths(
    cfg: dict[str, Any], tag: str
) -> list[str]:
    return list(
        {
            f"extra/output/pretrained_models/{tag}/{pretrain_name}/"
            f"dataset-{dataset}+instance_idx-{idx}/"
            f"pretrain_seed-{idx}/model.bundle"
            for pretrain_name in cfg["PRETRAIN_NAMES"]
            for method in cfg["METHODS_WITH_PRETRAINING_STAGE"]
            for dataset in cfg["DATASETS_USED_PER_METHOD"][method]
            for idx in cfg["DATASET_INSTANCE_INDICES"]
        }
    )


def _expected_trained_method_paths(cfg: dict[str, Any], tag: str) -> list[str]:
    no_pretrain = cfg["NO_PRETRAIN_STR"]
    paths = [
        f"extra/output/trained_methods/{tag}/{method}/"
        f"dataset-{dataset}+instance_idx-{idx}/"
        f"pretrain_seed-{idx}/"
        f"train_seed-{idx}+train_hard_budget-{tb}+train_soft_budget_param-{tsp}/method.bundle"
        for method in cfg["METHODS_WITH_PRETRAINING_STAGE"]
        for dataset in cfg["DATASETS"]
        for idx in cfg["DATASET_INSTANCE_INDICES"]
        for (tb, _eb, tsp, _esp) in cfg["BUDGET_PARAMS"][method][dataset]
    ]
    paths += [
        f"extra/output/trained_methods/{tag}/{method}/"
        f"dataset-{dataset}+instance_idx-{idx}/"
        f"{no_pretrain}/"
        f"train_seed-{idx}+train_hard_budget-{tb}+train_soft_budget_param-{tsp}/method.bundle"
        for method in cfg["METHODS_WITHOUT_PRETRAINING_STAGE"]
        for dataset in cfg["DATASETS"]
        for idx in cfg["DATASET_INSTANCE_INDICES"]
        for (tb, _eb, tsp, _esp) in cfg["BUDGET_PARAMS"][method][dataset]
    ]
    return paths


def _report(label: str, expected: list[str]) -> list[str]:
    missing = [p for p in expected if not Path(p).is_dir()]
    n_present = len(expected) - len(missing)
    print(f"  {label}: {n_present}/{len(expected)} present")
    return missing


def check_local(
    cfg: dict[str, Any], bundle_types: list[str], *, verbose: bool = False
) -> bool:
    """Check local completeness against expected pipeline outputs. Returns True if all present."""
    tag = f"initializer-{cfg['INITIALIZER']}"
    missing: list[str] = []

    if "trained_classifiers" in bundle_types:
        missing += _report("classifiers", _expected_classifier_paths(cfg, tag))
    if "pretrained_models" in bundle_types:
        missing += _report(
            "pretrained models", _expected_pretrained_model_paths(cfg, tag)
        )
    if "trained_methods" in bundle_types:
        missing += _report(
            "trained methods", _expected_trained_method_paths(cfg, tag)
        )

    if missing:
        print(f"{len(missing)} bundle(s) missing.")
        if verbose:
            for path in sorted(missing):
                print(f"  {path}")
        return False

    print("All expected bundles are present locally.")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that all bundles expected by the pipeline are present locally. "
            "Pass the same --configfile arguments as you would to Snakemake."
        )
    )
    parser.add_argument(
        "--configfile",
        action="append",
        required=True,
        type=Path,
        metavar="FILE",
        dest="configfiles",
        help="YAML config file (may be repeated, merged in order)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Override methods from configfiles (same as --config methods=[...] in Snakemake)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Override datasets from configfiles (same as --config datasets=[...] in Snakemake)",
    )
    parser.add_argument(
        "--instance-indices",
        nargs="+",
        type=int,
        default=[0, 1],
        metavar="IDX",
        help="Dataset instance indices (default: 0 1)",
    )
    parser.add_argument("--initializer", default="cold")
    parser.add_argument(
        "--bundle-types",
        nargs="+",
        default=list(_DEFAULT_BUNDLE_TYPES),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="List each missing bundle path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw = _load_and_merge_configs(args.configfiles)
    if args.methods is not None:
        raw["methods"] = args.methods
    if args.datasets is not None:
        raw["datasets"] = args.datasets
    raw["dataset_instance_indices"] = args.instance_indices
    raw["initializer"] = args.initializer
    cfg = load_config(raw)
    ok = check_local(cfg, args.bundle_types, verbose=args.verbose)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
