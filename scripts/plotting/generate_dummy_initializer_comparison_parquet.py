from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

PUBLICATION_METHODS = [
    "gadgil2023",
    "gadgil2023_ipw_feature_marginal",
    "aaco_full",
    "aaco_zero_fill",
    "aaco_mask_aware",
    "aaco_dr",
    "ol_with_mask",
]

DEFAULT_DATASETS = [
    "actg",
    "ckd",
    "cube",
    "cube_nm",
    "cube_nm_without_noise",
    "cube_without_noise",
    "diabetes",
    "physionet",
]

DEFAULT_MECHANISMS = [
    "mcar",
    "mar",
    "mnar_logistic",
]

DEFAULT_RATES = [0.3, 0.5, 0.7]

DATASET_BASE_METRICS = {
    "actg": 0.74,
    "ckd": 0.90,
    "cube": 0.69,
    "cube_nm": 0.42,
    "cube_nm_without_noise": 0.55,
    "cube_without_noise": 0.77,
    "diabetes": 0.79,
    "physionet": 0.57,
}

METHOD_BASE_OFFSETS = {
    "gadgil2023": 0.08,
    "gadgil2023_ipw_feature_marginal": 0.05,
    "aaco_full": 0.11,
    "aaco_zero_fill": 0.03,
    "aaco_mask_aware": 0.07,
    "aaco_dr": 0.10,
    "ol_with_mask": 0.00,
}

METHOD_RATE_PENALTIES = {
    "gadgil2023": {0.3: 0.020, 0.5: 0.035, 0.7: 0.055},
    "gadgil2023_ipw_feature_marginal": {0.3: 0.010, 0.5: 0.020, 0.7: 0.030},
    "aaco_full": {0.3: 0.000, 0.5: 0.000, 0.7: 0.000},
    "aaco_zero_fill": {0.3: 0.060, 0.5: 0.080, 0.7: 0.110},
    "aaco_mask_aware": {0.3: 0.025, 0.5: 0.035, 0.7: 0.045},
    "aaco_dr": {0.3: 0.012, 0.5: 0.018, 0.7: 0.028},
    "ol_with_mask": {0.3: 0.030, 0.5: 0.045, 0.7: 0.065},
}

MECHANISM_PENALTIES = {
    "mcar": 0.00,
    "mar": 0.02,
    "mnar_logistic": 0.04,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic merged eval parquet for tuning "
            "plot_eval_perf_by_initializer.py."
        )
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Where to write the dummy parquet.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to include in the synthetic benchmark.",
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=DEFAULT_MECHANISMS,
        help="Missingness mechanisms to include.",
    )
    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=DEFAULT_RATES,
        help="Missingness rates to include for non-cold initializers.",
    )
    parser.add_argument(
        "--n-train-seeds",
        type=int,
        default=3,
        help="Number of synthetic train seeds.",
    )
    parser.add_argument(
        "--n-eval-seeds",
        type=int,
        default=3,
        help="Number of synthetic eval seeds.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Number of examples per class in each synthetic run.",
    )
    parser.add_argument(
        "--eval-hard-budget",
        type=float,
        default=10.0,
        help="Hard budget value stored in the synthetic rows.",
    )
    parser.add_argument(
        "--train-hard-budget",
        type=float,
        default=10.0,
        help="Train hard budget value stored in the synthetic rows.",
    )
    return parser.parse_args()


def _rate_code(rate: float) -> str:
    rounded = round(rate * 10)
    return f"p{rounded:02d}"


def _initializer_name(mechanism: str | None, rate: float | None) -> str:
    if mechanism is None or rate is None:
        return "train_initializer-cold+eval_initializer-cold"
    return (
        f"train_initializer-{mechanism}_{_rate_code(rate)}"
        "+eval_initializer-cold"
    )


def _seed_adjustment(train_seed: int, eval_seed: int) -> float:
    centered_train = train_seed - 2
    centered_eval = eval_seed - 2
    return centered_train * 0.003 + centered_eval * 0.002


def _target_metric(
    dataset: str,
    afa_method: str,
    *,
    mechanism: str | None,
    rate: float | None,
    train_seed: int,
    eval_seed: int,
) -> float:
    base = DATASET_BASE_METRICS[dataset] + METHOD_BASE_OFFSETS[afa_method]
    if mechanism is not None and rate is not None:
        penalty = METHOD_RATE_PENALTIES[afa_method][rate]
        penalty += MECHANISM_PENALTIES[mechanism]
        base -= penalty
    metric = base + _seed_adjustment(train_seed, eval_seed)
    return min(max(metric, 0.35), 0.985)


def _binary_rows_for_metric(
    metric: float,
    *,
    n_per_class: int,
) -> list[tuple[int, int]]:
    max_errors_per_class = n_per_class
    errors_per_class = round((1.0 - metric) * n_per_class)
    errors_per_class = max(0, min(errors_per_class, max_errors_per_class))
    correct_per_class = n_per_class - errors_per_class
    return (
        [(0, 0)] * correct_per_class
        + [(0, 1)] * errors_per_class
        + [(1, 1)] * correct_per_class
        + [(1, 0)] * errors_per_class
    )


def main() -> None:
    args = parse_args()
    rows: list[dict[str, int | float | str | None]] = []

    initializers: list[tuple[str | None, float | None]] = [(None, None)]
    for mechanism in args.mechanisms:
        initializers.extend((mechanism, rate) for rate in args.rates)

    for dataset in args.datasets:
        if dataset not in DATASET_BASE_METRICS:
            msg = f"Unsupported dataset for dummy generator: {dataset}"
            raise ValueError(msg)
        for afa_method in PUBLICATION_METHODS:
            for mechanism, rate in initializers:
                initializer = _initializer_name(mechanism, rate)
                for train_seed in range(1, args.n_train_seeds + 1):
                    for eval_seed in range(1, args.n_eval_seeds + 1):
                        metric = _target_metric(
                            dataset,
                            afa_method,
                            mechanism=mechanism,
                            rate=rate,
                            train_seed=train_seed,
                            eval_seed=eval_seed,
                        )
                        confusion_rows = _binary_rows_for_metric(
                            metric,
                            n_per_class=args.samples_per_class,
                        )
                        for true_class, predicted_class in confusion_rows:
                            rows.append(
                                {
                                    "action_performed": 0,
                                    "predicted_class": predicted_class,
                                    "true_class": true_class,
                                    "dataset": dataset,
                                    "afa_method": afa_method,
                                    "initializer": initializer,
                                    "train_seed": train_seed,
                                    "eval_seed": eval_seed,
                                    "eval_hard_budget": args.eval_hard_budget,
                                    "train_hard_budget": args.train_hard_budget,
                                    "train_soft_budget_param": None,
                                    "eval_soft_budget_param": None,
                                }
                            )

    output_path = args.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(output_path)
    print(
        "Wrote synthetic initializer parquet "
        f"with {len(rows):,} rows to {output_path}"
    )


if __name__ == "__main__":
    main()
