import polars as pl
import torch

from afabench.common.custom_types import (
    Features,
    Label,
)
from afabench.eval.eval import process_batch
from afabench.test.helpers import (
    get_deterministic_action_fn,
    get_deterministic_afa_predict_fn,
    get_direct_unmask_fn,
    get_random_afa_predict_fn,
)


def process_batch_wrapper(
    actions: list[list[int]],
    features: Features | None = None,
    external_predictions: list[list[int]] | None = None,
    builtin_predictions: list[list[int]] | None = None,
    true_label: Label | None = None,
    selection_budget: float | None = None,
) -> pl.DataFrame:
    if features is None:
        features = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert features.ndim == 2, "Only 1D features with batch dim supported"
    n_samples = features.shape[0]
    n_features = features.shape[-1]

    if true_label is None:
        true_label = torch.zeros((n_samples, 4), dtype=torch.float32)
    n_classes = true_label.shape[-1]

    if external_predictions is None:
        external_afa_predict_fn = get_random_afa_predict_fn(
            n_classes=n_classes
        )
    else:
        external_afa_predict_fn = get_deterministic_afa_predict_fn(
            external_predictions, n_classes=n_classes
        )
    if builtin_predictions is None:
        builtin_afa_predict_fn = get_random_afa_predict_fn(n_classes=n_classes)
    else:
        builtin_afa_predict_fn = get_deterministic_afa_predict_fn(
            builtin_predictions, n_classes=n_classes
        )

    initial_feature_mask = torch.zeros_like(features, dtype=torch.bool)
    initial_masked_features = torch.zeros_like(features)
    n_selection_choices = n_features
    df = process_batch(
        # afa_action_fn=get_sequential_action_fn(),
        afa_action_fn=get_deterministic_action_fn(actions),
        afa_unmask_fn=get_direct_unmask_fn(),
        n_selection_choices=n_selection_choices,
        features=features,
        initial_feature_mask=initial_feature_mask,
        initial_masked_features=initial_masked_features,
        true_label=true_label,
        feature_shape=torch.Size((n_features,)),
        external_afa_predict_fn=external_afa_predict_fn,
        builtin_afa_predict_fn=builtin_afa_predict_fn,
        selection_budget=selection_budget,
        selection_costs=None,
    )
    df = pl.from_pandas(df)
    return df


def add_time_column(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(time=pl.col("prev_selections_performed").list.len())


def assert_predictions(
    df: pl.DataFrame,
    idx: int,
    expected_predictions: list[int],
    prediction_type: str,
) -> None:
    if prediction_type == "external":
        prediction_col = "external_predicted_class"
    elif prediction_type == "builtin":
        prediction_col = "builtin_predicted_class"
    else:
        raise ValueError
    predictions = df.filter(pl.col("idx") == idx).sort("time")[prediction_col]
    assert (predictions == expected_predictions).all(), (
        f"Expected {predictions.to_list()} and {expected_predictions} to be equal."
    )


def test_expected_length() -> None:
    """Test that the returned dataframe has an expected number of rows."""
    features = torch.tensor([[1, 2, 3], [4, 5, 6]])
    actions = [[1, 2, 3, 0], [1, 2, 3, 0]]

    df = process_batch_wrapper(features=features, actions=actions)

    # With 3 features, we should have 4 rows for each sample. We make one prediction at 0 features, 1 feature, 2 features, and 3 features
    assert len(df.filter(pl.col("idx") == 0)) == 4
    assert len(df.filter(pl.col("idx") == 1)) == 4
    assert len(df) == 8


def test_external_predictions() -> None:
    # Batched
    features = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    actions = [[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]]
    external_predictions = [[0, 1, 3, 2, 1], [3, 1, 0, 2, 0]]

    df = process_batch_wrapper(
        features=features,
        actions=actions,
        external_predictions=external_predictions,
    )
    df = add_time_column(df)

    assert_predictions(
        df,
        idx=0,
        expected_predictions=external_predictions[0],
        prediction_type="external",
    )

    assert_predictions(
        df,
        idx=1,
        expected_predictions=external_predictions[1],
        prediction_type="external",
    )


def test_builtin_predictions() -> None:
    # Batched
    features = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    actions = [[1, 2, 3, 4, 0], [1, 2, 3, 4, 0]]
    builtin_predictions = [[0, 1, 3, 2, 1], [3, 1, 0, 2, 0]]

    df = process_batch_wrapper(
        features=features,
        actions=actions,
        builtin_predictions=builtin_predictions,
    )
    df = add_time_column(df)

    assert_predictions(
        df,
        idx=0,
        expected_predictions=builtin_predictions[0],
        prediction_type="builtin",
    )
    assert_predictions(
        df,
        idx=1,
        expected_predictions=builtin_predictions[1],
        prediction_type="builtin",
    )
