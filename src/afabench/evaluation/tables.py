"""
Load plotting-ready evaluation tables and prepare them for plots.

A plotting-ready table is the output of the transformation stage, merged
over runs and usually split by classifier type
(`docs/reference/evaluation_dataframes.md`). Every visualization job reads
it through `load_plotting_ready_table`, so all plots see the same dtypes.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from afabench.evaluation.schemas import PLOTTING_READY_DTYPES


def _cast_column(series: pd.Series, dtype: str) -> pd.Series:
    """Cast a column, turning non-numeric values of numeric dtypes to null."""
    if dtype in {"string", "boolean", "object"}:
        return series.astype(dtype)
    return cast("pd.Series", pd.to_numeric(series, errors="coerce")).astype(
        dtype
    )


def load_plotting_ready_table(path: Path) -> pd.DataFrame:
    """
    Read a plotting-ready evaluation table with nullable pandas dtypes.

    Merging tables whose column types disagree upcasts those columns to
    strings, so each known column is cast back to its dtype. In a numeric
    column, values that do not parse as numbers become null; a number the
    dtype cannot hold, such as a fraction in a `UInt64` column, and a
    boolean column value that is not bool-like raise `TypeError`. Absent
    columns stay absent.
    """
    table = pd.read_parquet(path, dtype_backend="numpy_nullable")
    for name, dtype in PLOTTING_READY_DTYPES.items():
        if name in table.columns:
            table[name] = _cast_column(table.loc[:, name], dtype)
    return table


class SoftBudgetParamConflictError(ValueError):
    """Raised when a row sets both the training and evaluation parameter."""


def combine_soft_budget_params(table: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the two soft-budget parameter columns by `soft_budget_param`.

    A row may set the training or the evaluation soft-budget parameter, never
    both; `soft_budget_param` holds whichever is set.
    """
    train = table["train_soft_budget_param"]
    evaluation = table["eval_soft_budget_param"]
    both_set = (train.notna() & evaluation.notna()).to_numpy(dtype=bool)
    if both_set.any():
        row = table.loc[both_set].iloc[0]
        msg = (
            "A row sets both soft-budget parameters, "
            f"train_soft_budget_param={row['train_soft_budget_param']} and "
            f"eval_soft_budget_param={row['eval_soft_budget_param']}; "
            "expected at most one."
        )
        raise SoftBudgetParamConflictError(msg)
    return table.assign(soft_budget_param=train.fillna(evaluation)).drop(
        columns=["train_soft_budget_param", "eval_soft_budget_param"]
    )


def keep_largest_hard_budget(table: pd.DataFrame) -> pd.DataFrame:
    """For each dataset, keep only the rows of its largest hard budget."""
    largest = table.groupby("dataset")["eval_hard_budget"].transform("max")
    return table.loc[(table["eval_hard_budget"] == largest).fillna(False)]


def keep_smallest_soft_budget_param(table: pd.DataFrame) -> pd.DataFrame:
    """For each method and dataset, keep only its smallest soft-budget parameter."""
    smallest = table.groupby(["afa_method", "dataset"])[
        "soft_budget_param"
    ].transform("min")
    return table.loc[(table["soft_budget_param"] == smallest).fillna(False)]


def _probability_matrix(series: pd.Series) -> NDArray[np.float64]:
    vectors: list[NDArray[np.float64]] = []
    for value in cast("list[object]", series.tolist()):
        if not isinstance(value, (list, np.ndarray)):
            msg = (
                f"predicted_probabilities must contain vectors, got {value!r}"
            )
            raise TypeError(msg)
        vector = np.asarray(cast("ArrayLike", value))
        if (
            vector.ndim != 1
            or vector.size < 2
            or (vectors and vector.shape != vectors[0].shape)
        ):
            msg = (
                "predicted_probabilities must contain equally sized class "
                f"vectors with at least two classes, got {value!r}"
            )
            raise ValueError(msg)
        if not np.issubdtype(vector.dtype, np.floating):
            msg = (
                "predicted_probabilities must contain floating values, "
                f"got {value!r} with dtype {vector.dtype}"
            )
            raise TypeError(msg)
        if not np.isfinite(vector).all():
            msg = (
                "predicted_probabilities must contain finite values, "
                f"got {value!r}"
            )
            raise ValueError(msg)
        if ((vector < 0) | (vector > 1)).any():
            msg = f"predicted_probabilities must be in [0, 1], got {value!r}"
            raise ValueError(msg)
        if not np.isclose(vector.sum(), 1.0, atol=1e-6, rtol=1e-5):
            msg = (
                "predicted_probabilities must sum to 1, "
                f"got {value!r} with sum {vector.sum()}"
            )
            raise ValueError(msg)
        vectors.append(vector.astype(np.float64))
    return np.stack(vectors)


def _classification_metrics(group: pd.DataFrame) -> pd.Series:
    true_class = group["true_class"].to_numpy(dtype=np.int64)
    probabilities = _probability_matrix(
        cast("pd.Series", group["predicted_probabilities"])
    )
    # Accuracy and F1 decide the most probable class, the lowest on ties.
    predicted_class = probabilities.argmax(axis=1)
    return pd.Series(
        {
            "accuracy": accuracy_score(true_class, predicted_class),
            "f_score": f1_score(true_class, predicted_class, average="macro"),
            "auroc": (
                roc_auc_score(true_class, probabilities[:, 1])
                if probabilities.shape[1] == 2
                # Binary OVR scores avoid sklearn's stricter multiclass
                # sum check; validation uses the evaluator's tolerance.
                else np.mean(
                    [
                        cast(
                            "float",
                            roc_auc_score(
                                true_class == class_index,
                                probabilities[:, class_index],
                            ),
                        )
                        for class_index in range(probabilities.shape[1])
                    ]
                )
            ),
        }
    )


# A run: one method trained and evaluated with one pair of seeds and budgets
RUN_COLUMNS = [
    "afa_method",
    "dataset",
    "train_seed",
    "eval_seed",
    "eval_hard_budget",
    "soft_budget_param",
]


def metrics_at_stop(table: pd.DataFrame) -> pd.DataFrame:
    """
    Score each run on its rows, which should be its stop actions.

    Columns: the run columns, `accuracy`, `f_score` (macro F1), `auroc`
    and `avg_accumulated_cost`. Class probabilities use label index order.
    """
    grouped = table.groupby(RUN_COLUMNS, dropna=False)
    metrics = grouped[["true_class", "predicted_probabilities"]].apply(
        _classification_metrics
    )
    metrics["avg_accumulated_cost"] = grouped["accumulated_cost"].mean()
    return metrics.reset_index()


def metrics_at_every_step(table: pd.DataFrame) -> pd.DataFrame:
    """Score each run at each time step (`n_selections_performed`)."""
    return (
        table.groupby([*RUN_COLUMNS, "n_selections_performed"], dropna=False)[
            ["true_class", "predicted_probabilities"]
        ]
        .apply(_classification_metrics)
        .reset_index()
    )


def variance_over_seeds_at_stop(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize stop metrics over seeds; undefined AUROCs propagate."""
    return (
        metrics.groupby(
            ["afa_method", "dataset", "eval_hard_budget", "soft_budget_param"],
            dropna=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f_score=("f_score", "mean"),
            std_f_score=("f_score", "std"),
            mean_auroc=("auroc", lambda values: values.mean(skipna=False)),
            std_auroc=("auroc", lambda values: values.std(skipna=False)),
            mean_avg_accumulated_cost=("avg_accumulated_cost", "mean"),
            std_avg_accumulated_cost=("avg_accumulated_cost", "std"),
        )
        .reset_index()
    )


def variance_over_seeds_at_every_step(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-step metrics over seeds; undefined AUROCs propagate."""
    return (
        metrics.groupby(
            [
                "afa_method",
                "dataset",
                "eval_hard_budget",
                "soft_budget_param",
                "n_selections_performed",
            ],
            dropna=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_f_score=("f_score", "mean"),
            std_f_score=("f_score", "std"),
            mean_auroc=("auroc", lambda values: values.mean(skipna=False)),
            std_auroc=("auroc", lambda values: values.std(skipna=False)),
        )
        .reset_index()
    )


# (method, dataset) -> the sets of methods beside which the method is kept on
# that dataset; beside any other set of methods it is excluded
type Exclusions = Mapping[tuple[str, str], frozenset[frozenset[str]]]

EXCLUSION_MAPPING: Exclusions = {}


def apply_exclusions(
    table: pd.DataFrame, exclusions: Exclusions = EXCLUSION_MAPPING
) -> pd.DataFrame:
    """
    Drop each excluded method from a dataset unless allowed by its methods.

    For example, the exclusion

        ("odin_model_based", "diabetes"): frozenset(
            [frozenset(["odin_model_based", "odin_model_free"])]
        )

    keeps odin_model_based on diabetes only when the methods on diabetes are
    exactly odin_model_based and odin_model_free.
    """
    method_dataset_pairs = (
        table[["afa_method", "dataset"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for method, dataset in method_dataset_pairs:
        allowed_method_sets = exclusions.get((method, dataset))
        if allowed_method_sets is None:
            continue
        methods_on_dataset = frozenset(
            table.loc[table["dataset"] == dataset, "afa_method"].unique()
        )
        if methods_on_dataset not in allowed_method_sets:
            table = table.loc[
                ~(
                    (table["afa_method"] == method)
                    & (table["dataset"] == dataset)
                )
            ]
    return table
