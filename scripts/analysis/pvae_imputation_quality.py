# pyright: reportCallIssue=false, reportAttributeAccessIssue=false
"""
Measure PVAE imputation quality on mechanism-missing entries.

Regenerates the training missingness mask of an initializer (same seeding
convention as AACO/RL/pretraining: `initializer.set_seed(seed)` followed by
one initialize over the full matrix), imputes the missing entries with a
pretrained PVAE (label-free and label-conditioned), and reports RMSE against
the true hidden values. This is possible because missingness is synthetic.

The output parquet is meant to be joined with merged eval-perf results on
(dataset, initializer/mechanism, rate, seed) to correlate policy degradation
with generator error (H3).

Usage:
    python scripts/analysis/pvae_imputation_quality.py \
        --dataset-bundle extra/output/datasets/<dataset>/<idx>/train.bundle \
        --pvae-bundle extra/output/pretrained_models/initializer-<init>/pvae_missing/.../model.bundle \
        --initializer mcar_p05 \
        --seed 0 \
        --tag honest \
        --output extra/output/analysis/pvae_quality/<...>.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
import torch
import yaml

from afabench.common.bundle import load_bundle
from afabench.common.config_classes import InitializerConfig
from afabench.common.initializers.utils import (
    get_afa_initializer_from_config,
)
from afabench.missing_values.restoration import (
    derive_train_support_masks,
    load_pvae_model,
    restore_missing_features_with_pvae,
)

if TYPE_CHECKING:
    from afabench.common.custom_types import AFADataset

INITIALIZER_CONF_DIR = Path("extra/conf/components/initializers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-bundle", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name recorded in the output (for joining with "
        "eval-perf results). Defaults to the bundle's parent directory name.",
    )
    parser.add_argument("--pvae-bundle", type=Path, required=True)
    parser.add_argument(
        "--initializer",
        required=True,
        help=(
            "Initializer config name under "
            f"{INITIALIZER_CONF_DIR} (e.g. mcar_p05)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--tag",
        default="honest",
        help="Free-form generator tag recorded in the output "
        "(e.g. honest, oracle_gen).",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _rmse_rows(
    *,
    restored: torch.Tensor,
    features: torch.Tensor,
    missing: torch.Tensor,
    label_conditioning: bool,
    base_row: dict,
) -> list[dict]:
    rows: list[dict] = []
    squared_errors = (restored - features) ** 2

    overall_n = int(missing.sum().item())
    overall_rmse = (
        squared_errors[missing].mean().sqrt().item() if overall_n else None
    )
    rows.append(
        base_row
        | {
            "label_conditioning": label_conditioning,
            "feature_index": None,
            "n_missing": overall_n,
            "rmse": overall_rmse,
        }
    )

    for feature_index in range(features.shape[1]):
        feature_missing = missing[:, feature_index]
        n_missing = int(feature_missing.sum().item())
        rmse = (
            squared_errors[feature_missing, feature_index].mean().sqrt().item()
            if n_missing
            else None
        )
        rows.append(
            base_row
            | {
                "label_conditioning": label_conditioning,
                "feature_index": feature_index,
                "n_missing": n_missing,
                "rmse": rmse,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    initializer_conf_path = INITIALIZER_CONF_DIR / f"{args.initializer}.yaml"
    initializer_conf = yaml.safe_load(initializer_conf_path.read_text())
    initializer = get_afa_initializer_from_config(
        InitializerConfig(**initializer_conf)
    )

    dataset_obj, _manifest = load_bundle(Path(args.dataset_bundle))
    dataset = cast("AFADataset", cast("object", dataset_obj))
    features, labels = dataset.get_all_data()

    _observed_mask, train_support_mask = derive_train_support_masks(
        initializer,
        seed=args.seed,
        features=features,
        feature_shape=dataset.feature_shape,
    )
    flat_features = features.reshape(features.shape[0], -1).to(device)
    flat_support = cast(
        "torch.BoolTensor",
        train_support_mask.reshape(features.shape[0], -1).to(device),
    )
    missing = ~flat_support

    pvae_model = load_pvae_model(Path(args.pvae_bundle), device=device)

    dataset_name = (
        args.dataset
        if args.dataset is not None
        else args.dataset_bundle.parent.parent.name
    )
    base_row = {
        "dataset": dataset_name,
        "dataset_bundle": str(args.dataset_bundle),
        "pvae_bundle": str(args.pvae_bundle),
        "pvae_tag": args.tag,
        "initializer": args.initializer,
        "seed": args.seed,
        "missing_fraction": missing.float().mean().item(),
    }

    rows: list[dict] = []
    for label_conditioning in (False, True):
        restored = restore_missing_features_with_pvae(
            flat_features,
            flat_support,
            pvae_model=pvae_model,
            label=labels.to(device).float() if label_conditioning else None,
        )
        rows.extend(
            _rmse_rows(
                restored=restored.cpu(),
                features=flat_features.cpu(),
                missing=missing.cpu(),
                label_conditioning=label_conditioning,
                base_row=base_row,
            )
        )

    result_df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(args.output)
    overall = result_df[result_df["feature_index"].isna()]
    print(overall.to_string(index=False))
    print(f"Wrote {len(result_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
