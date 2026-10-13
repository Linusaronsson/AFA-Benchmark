"""Plotting-ready evaluation tables load and filter the same way for every plot."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from afabench.evaluation.tables import (
    apply_exclusions,
    combine_soft_budget_params,
    keep_largest_hard_budget,
    keep_smallest_soft_budget_param,
    load_plotting_ready_table,
    metrics_at_every_step,
    metrics_at_stop,
    variance_over_seeds_at_every_step,
    variance_over_seeds_at_stop,
)


def test_loaded_table_has_nullable_dtypes_and_nulls_for_upcast_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval_perf.parquet"
    # Merging tables whose column types disagree upcasts columns to strings
    pq.write_table(
        pa.table(
            {
                "action_performed": pa.array([0, 2], pa.int64()),
                "true_class": pa.array([1, 0], pa.int64()),
                "accumulated_cost": pa.array([1.0, 2.0], pa.float64()),
                "forced_stop": pa.array([False, True]),
                "n_selections_performed": pa.array([1, 2], pa.int64()),
                "predicted_probabilities": pa.array(
                    [[0.25, 0.75], None], pa.list_(pa.float64())
                ),
                "afa_method": pa.array(["jafa", "jafa"]),
                "dataset": pa.array(["cube", "cube"]),
                "initializer": pa.array(["cold", "cold"]),
                "train_seed": pa.array([0, 0], pa.int64()),
                "eval_seed": pa.array([0, 0], pa.int64()),
                "eval_hard_budget": pa.array(["2.0", "None"]),
                "eval_soft_budget_param": pa.array([None, None], pa.float64()),
            }
        ),
        path,
    )

    table = load_plotting_ready_table(path)

    assert {name: str(dtype) for name, dtype in table.dtypes.items()} == {
        "action_performed": "UInt64",
        "true_class": "UInt64",
        "accumulated_cost": "Float64",
        "forced_stop": "boolean",
        "n_selections_performed": "UInt64",
        "predicted_probabilities": "object",
        "afa_method": "string",
        "dataset": "string",
        "initializer": "string",
        "train_seed": "UInt64",
        "eval_seed": "UInt64",
        "eval_hard_budget": "Float64",
        "eval_soft_budget_param": "Float64",
    }
    assert table["eval_hard_budget"].tolist() == [2.0, pd.NA]
    assert table["predicted_probabilities"].iloc[0].tolist() == [0.25, 0.75]
    assert table["predicted_probabilities"].iloc[1] is None


def budget_rows(
    rows: list[tuple[str, str, float | None, float | None]],
) -> pd.DataFrame:
    """Rows of (afa_method, dataset, eval_hard_budget, soft_budget_param)."""
    return pd.DataFrame(
        rows,
        columns=pd.Index(
            ["afa_method", "dataset", "eval_hard_budget", "soft_budget_param"]
        ),
    ).astype(
        {
            "afa_method": "string",
            "dataset": "string",
            "eval_hard_budget": "Float64",
            "soft_budget_param": "Float64",
        }
    )


def test_keep_largest_hard_budget_keeps_each_datasets_largest() -> None:
    table = budget_rows(
        [
            ("jafa", "cube", 1.0, None),
            ("gdfs", "cube", 3.0, None),
            ("jafa", "mnist", 2.0, None),
            # Soft-budget rows have no hard budget to compare
            ("jafa", "mnist", None, 0.1),
        ]
    )

    kept = keep_largest_hard_budget(table)

    assert kept.to_numpy().tolist() == [
        ["gdfs", "cube", 3.0, pd.NA],
        ["jafa", "mnist", 2.0, pd.NA],
    ]


def test_keep_smallest_soft_budget_param_per_method_and_dataset() -> None:
    table = budget_rows(
        [
            ("jafa", "cube", None, 0.1),
            ("jafa", "cube", None, 0.01),
            ("gdfs", "cube", None, 0.5),
            ("jafa", "mnist", None, 1.0),
            ("jafa", "mnist", 2.0, None),
        ]
    )

    kept = keep_smallest_soft_budget_param(table)

    assert kept.to_numpy().tolist() == [
        ["jafa", "cube", pd.NA, 0.01],
        ["gdfs", "cube", pd.NA, 0.5],
        ["jafa", "mnist", pd.NA, 1.0],
    ]


def soft_budget_params(
    train: list[float | None], evaluation: list[float | None]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "train_soft_budget_param": train,
            "eval_soft_budget_param": evaluation,
        }
    ).astype("Float64")


def test_combine_soft_budget_params_takes_whichever_is_set() -> None:
    table = soft_budget_params([0.1, None, None], [None, 0.5, None])

    combined = combine_soft_budget_params(table)

    assert combined.columns.tolist() == ["soft_budget_param"]
    assert combined["soft_budget_param"].tolist() == [0.1, 0.5, pd.NA]


def test_combine_soft_budget_params_rejects_rows_with_both_set() -> None:
    table = soft_budget_params([0.1, 0.2], [None, 0.5])

    with pytest.raises(ValueError, match=r"train_soft_budget_param=0\.2"):
        combine_soft_budget_params(table)


@pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.UndefinedMetricWarning"
)
def test_metrics_at_stop_score_each_seed_and_budget() -> None:
    table = pd.DataFrame(
        {
            "afa_method": ["jafa"] * 3,
            "dataset": ["cube"] * 3,
            "train_seed": [0, 0, 1],
            "eval_seed": [0, 0, 0],
            "eval_hard_budget": [2.0, 2.0, 2.0],
            "soft_budget_param": [None, None, None],
            "true_class": [1, 0, 1],
            "predicted_probabilities": [[0.2, 0.8]] * 3,
            "accumulated_cost": [1.0, 2.0, 2.0],
        }
    ).astype({"soft_budget_param": "Float64"})

    metrics = metrics_at_stop(table)

    assert metrics.to_dict("records") == [
        {
            "afa_method": "jafa",
            "dataset": "cube",
            "train_seed": 0,
            "eval_seed": 0,
            "eval_hard_budget": 2.0,
            "soft_budget_param": None,
            "accuracy": 0.5,
            # Macro F1 of classes 0 (F1 0) and 1 (F1 2/3)
            "f_score": pytest.approx(1 / 3),
            "auroc": 0.5,
            "avg_accumulated_cost": 1.5,
        },
        {
            "afa_method": "jafa",
            "dataset": "cube",
            "train_seed": 1,
            "eval_seed": 0,
            "eval_hard_budget": 2.0,
            "soft_budget_param": None,
            "accuracy": 1.0,
            "f_score": 1.0,
            "auroc": pytest.approx(float("nan"), nan_ok=True),
            "avg_accumulated_cost": 2.0,
        },
    ]


def test_excluded_method_is_kept_only_beside_its_allowed_methods() -> None:
    exclusions = {
        ("odin_model_based", "diabetes"): frozenset(
            [frozenset(["odin_model_based", "odin_model_free"])]
        )
    }
    odin_pair = budget_rows(
        [
            ("odin_model_based", "diabetes", 1.0, None),
            ("odin_model_free", "diabetes", 1.0, None),
        ]
    )
    with_jafa = budget_rows(
        [
            ("odin_model_based", "diabetes", 1.0, None),
            ("jafa", "diabetes", 1.0, None),
            ("odin_model_based", "cube", 1.0, None),
        ]
    )

    assert apply_exclusions(odin_pair, exclusions).equals(odin_pair)
    assert apply_exclusions(with_jafa, exclusions)[
        ["afa_method", "dataset"]
    ].to_numpy().tolist() == [
        ["jafa", "diabetes"],
        ["odin_model_based", "cube"],
    ]


def test_loaded_table_preserves_probability_vectors_and_missing_classifiers(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval_perf.parquet"
    pd.DataFrame(
        {"predicted_probabilities": [[0.1, 0.2, 0.7], None, [0.8, 0.1, 0.1]]}
    ).to_parquet(path)

    table = load_plotting_ready_table(path)

    assert str(table["predicted_probabilities"].dtype) == "object"
    assert table["predicted_probabilities"].iloc[0].tolist() == [0.1, 0.2, 0.7]
    assert pd.isna(table["predicted_probabilities"].iloc[1])
    assert table["predicted_probabilities"].iloc[2].tolist() == [0.8, 0.1, 0.1]


def classification_rows(
    true_classes: list[int], probabilities: list[list[float]]
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "afa_method": "jafa",
            "dataset": "cube",
            "train_seed": 0,
            "eval_seed": 0,
            "eval_hard_budget": 2.0,
            "soft_budget_param": None,
            "n_selections_performed": 0,
            "true_class": true_classes,
            "predicted_probabilities": probabilities,
            "accumulated_cost": 1.0,
        }
    ).astype({"soft_budget_param": "Float64"})


def test_class_metrics_decide_the_most_probable_class_lowest_on_ties() -> None:
    # Argmax decides [1, 0, 0]; the tie on the last row goes to class 0.
    table = classification_rows(
        [1, 0, 0], [[0.4, 0.6], [0.7, 0.3], [0.5, 0.5]]
    )

    metrics = metrics_at_stop(table)

    assert metrics[["accuracy", "f_score"]].to_numpy().tolist() == [[1.0, 1.0]]


def test_metrics_at_stop_binary_auroc_uses_class_one_probabilities() -> None:
    table = classification_rows(
        [0, 0, 1, 1], [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]
    )

    metrics = metrics_at_stop(table)

    # Three of the four positive-negative pairs are ordered correctly.
    assert metrics["auroc"].tolist() == [0.75]
    assert metrics["accuracy"].tolist() == [0.75]
    assert metrics["f_score"].tolist() == [pytest.approx(11 / 15)]
    assert metrics["avg_accumulated_cost"].tolist() == [1.0]


def test_metrics_at_stop_multiclass_auroc_is_macro_one_vs_rest() -> None:
    table = classification_rows(
        [0, 1, 2], [[0.4, 0.5, 0.1], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]]
    )

    metrics = metrics_at_stop(table)

    # Class 0: 1; class 1: 1/2; class 2: 1/2 (ties). Macro mean: 2/3.
    assert metrics["auroc"].tolist() == [pytest.approx(2 / 3)]
    assert metrics["accuracy"].tolist() == [pytest.approx(1 / 3)]
    assert metrics["f_score"].tolist() == [pytest.approx(1 / 6)]


@pytest.mark.parametrize(
    ("true_classes", "probabilities"),
    [
        ([0, 0], [[0.9, 0.1], [0.8, 0.2]]),
        ([1, 1], [[0.1, 0.9], [0.2, 0.8]]),
        ([0, 1], [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]),
        ([1, 2], [[0.1, 0.8, 0.1], [0.1, 0.2, 0.7]]),
    ],
    ids=[
        "no_class_one",
        "no_class_zero",
        "no_class_two",
        "no_class_zero_multi",
    ],
)
def test_metrics_at_stop_missing_required_classes_have_unavailable_auroc(
    true_classes: list[int], probabilities: list[list[float]]
) -> None:
    table = classification_rows(true_classes, probabilities)

    metrics = metrics_at_stop(table)

    assert metrics["auroc"].isna().all()
    assert metrics["accuracy"].tolist() == [1.0]
    assert metrics["f_score"].tolist() == [1.0]


def test_metrics_at_every_step_score_probabilities_within_each_run_and_step() -> (
    None
):
    initial = classification_rows(
        [0, 0, 1, 1], [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]
    )
    later = classification_rows([1, 1], [[0.1, 0.9], [0.2, 0.8]]).assign(
        n_selections_performed=1
    )
    other_seed = classification_rows([0, 1], [[0.1, 0.9], [0.9, 0.1]]).assign(
        train_seed=1
    )

    metrics = metrics_at_every_step(pd.concat([initial, later, other_seed]))

    assert metrics[
        ["train_seed", "n_selections_performed"]
    ].to_numpy().tolist() == [
        [0, 0],
        [0, 1],
        [1, 0],
    ]
    assert metrics["auroc"].iloc[0] == 0.75
    assert pd.isna(metrics["auroc"].iloc[1])
    assert metrics["auroc"].iloc[2] == 0.0
    assert metrics["accuracy"].tolist() == [0.75, 1.0, 0.0]


@pytest.mark.parametrize(
    "every_step", [False, True], ids=["stop", "every_step"]
)
def test_auroc_seed_summaries_average_run_scores_not_pooled_predictions(
    every_step: bool,
) -> None:
    first = classification_rows(
        [0, 0, 1, 1], [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]
    )
    second = classification_rows(
        [0, 0, 1, 1], [[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]]
    ).assign(train_seed=1, eval_seed=1)
    table = pd.concat([first, second])

    if every_step:
        summary = variance_over_seeds_at_every_step(
            metrics_at_every_step(table)
        )
    else:
        summary = variance_over_seeds_at_stop(metrics_at_stop(table))

    # Per-run AUROCs 3/4 and 1/4; sample std uses ddof=1, as other metrics do.
    assert summary["mean_auroc"].tolist() == [0.5]
    assert summary["std_auroc"].tolist() == [pytest.approx(0.3535533905932738)]
    assert summary["mean_accuracy"].tolist() == [0.5]


@pytest.mark.parametrize(
    "every_step", [False, True], ids=["stop", "every_step"]
)
def test_auroc_seed_summaries_propagate_undefined_runs(
    every_step: bool,
) -> None:
    defined = classification_rows(
        [0, 0, 1, 1], [[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]]
    )
    undefined = classification_rows([1, 1], [[0.1, 0.9], [0.2, 0.8]]).assign(
        train_seed=1, eval_seed=1
    )
    table = pd.concat(
        [defined, undefined, defined.assign(train_seed=2, eval_seed=2)]
    )

    if every_step:
        summary = variance_over_seeds_at_every_step(
            metrics_at_every_step(table)
        )
    else:
        summary = variance_over_seeds_at_stop(metrics_at_stop(table))

    assert summary["mean_auroc"].isna().all()
    assert summary["std_auroc"].isna().all()
    # Other metrics keep their existing seed means.
    assert summary["mean_accuracy"].tolist() == [pytest.approx(5 / 6)]


@pytest.mark.parametrize(
    ("probabilities", "error"),
    [
        ([None, [0.2, 0.8]], TypeError),
        ([0.5, [0.2, 0.8]], TypeError),
        ([[1.0], [1.0]], ValueError),
        ([[], []], ValueError),
        ([[[0.8, 0.2]], [[0.2, 0.8]]], ValueError),
        ([[0.8, 0.2], [0.2, 0.3, 0.5]], ValueError),
    ],
    ids=["null", "scalar", "one_class", "empty", "nested", "different_widths"],
)
def test_metrics_reject_malformed_probability_vectors(
    probabilities: list[object], error: type[ValueError] | type[TypeError]
) -> None:
    table = classification_rows([0, 1], [[0.8, 0.2], [0.2, 0.8]])
    table["predicted_probabilities"] = probabilities

    with pytest.raises(error, match=r"predicted_probabilities.*vector"):
        metrics_at_stop(table)


@pytest.mark.parametrize("vector", [[1, 0], ["0.8", "0.2"], [True, False]])
def test_metrics_require_floating_probability_values(
    vector: list[object],
) -> None:
    table = classification_rows([0, 1], [[0.8, 0.2], [0.2, 0.8]])
    table["predicted_probabilities"] = [vector, [0.2, 0.8]]

    with pytest.raises(TypeError, match=r"predicted_probabilities.*floating"):
        metrics_at_stop(table)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_metrics_reject_nonfinite_probabilities(value: float) -> None:
    table = classification_rows([0, 1], [[0.8, 0.2], [0.2, 0.8]])
    table["predicted_probabilities"] = [[value, 0.2], [0.2, 0.8]]

    with pytest.raises(ValueError, match=r"predicted_probabilities.*finite"):
        metrics_at_stop(table)


@pytest.mark.parametrize("vector", [[-0.1, 1.1], [1.1, -0.1]])
def test_metrics_reject_probabilities_outside_unit_interval(
    vector: list[float],
) -> None:
    table = classification_rows([0, 1], [[0.8, 0.2], [0.2, 0.8]])
    table["predicted_probabilities"] = [vector, [0.2, 0.8]]

    with pytest.raises(ValueError, match=r"predicted_probabilities.*\[0, 1\]"):
        metrics_at_stop(table)


def test_metrics_reject_unnormalized_probabilities_without_inferring_logits() -> (
    None
):
    table = classification_rows([0, 1], [[0.8, 0.2], [0.2, 0.8]])
    table["predicted_probabilities"] = [[0.6, 0.2], [0.2, 0.8]]

    with pytest.raises(ValueError, match=r"predicted_probabilities.*sum to 1"):
        metrics_at_stop(table)


def test_metrics_accept_multiclass_probabilities_with_evaluation_sum_tolerance() -> (
    None
):
    # Evaluation allows row sums within atol=1e-6, rtol=1e-5 of one.
    table = classification_rows(
        [0, 1, 2],
        [[0.4, 0.5, 0.0999895], [0.3, 0.6, 0.0999895], [0.2, 0.7, 0.0999895]],
    )

    metrics = metrics_at_stop(table)

    assert metrics["auroc"].tolist() == [pytest.approx(2 / 3)]


def test_metrics_score_probability_arrays_reloaded_from_parquet(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval_perf.parquet"
    classification_rows(
        [0, 1, 2], [[0.4, 0.5, 0.1], [0.3, 0.6, 0.1], [0.2, 0.7, 0.1]]
    ).to_parquet(path)

    metrics = metrics_at_stop(load_plotting_ready_table(path))

    assert metrics["auroc"].tolist() == [pytest.approx(2 / 3)]


@pytest.mark.parametrize(
    "every_step", [False, True], ids=["stop", "every_step"]
)
@pytest.mark.parametrize(
    "defined", [False, True], ids=["undefined", "defined"]
)
def test_auroc_single_seed_summaries_have_unavailable_standard_deviation(
    every_step: bool, defined: bool
) -> None:
    table = (
        classification_rows([0, 1], [[0.8, 0.2], [0.2, 0.8]])
        if defined
        else classification_rows([1, 1], [[0.1, 0.9], [0.2, 0.8]])
    )

    if every_step:
        summary = variance_over_seeds_at_every_step(
            metrics_at_every_step(table)
        )
    else:
        summary = variance_over_seeds_at_stop(metrics_at_stop(table))

    if defined:
        assert summary["mean_auroc"].tolist() == [1.0]
    else:
        assert summary["mean_auroc"].isna().all()
    assert summary["std_auroc"].isna().all()
