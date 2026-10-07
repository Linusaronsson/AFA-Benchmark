"""Contracts for evaluation rows and metadata-enriched Parquet results."""

from numbers import Integral, Real
from typing import Any, Literal

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from afabench.evaluation.history import validate_episode_log


class BatchEvaluationSchema(pa.DataFrameModel):
    """One row per episode and zero-based time step of one evaluated batch."""

    episode_id: Series[int] = pa.Field(ge=0)
    step: Series[int] = pa.Field(ge=0)
    action_performed: Series[int] = pa.Field(ge=0)
    # Predictions use int64 when supplied and object/None otherwise. Avoid
    # coercion so existing callers and Parquet artifacts keep their dtypes.
    builtin_predicted_class: Series[Any] = pa.Field(nullable=True)
    external_predicted_class: Series[Any] = pa.Field(nullable=True)
    true_class: Series[int] = pa.Field(ge=0)
    accumulated_cost: Series[float] = pa.Field(ge=0)
    forced_stop: Series[bool] = pa.Field()

    @pa.dataframe_check
    @classmethod
    def complete_episodes(cls, frame: pd.DataFrame) -> bool:
        try:
            validate_episode_log(frame)
        except ValueError:
            return False
        return True

    @pa.check(
        "builtin_predicted_class",
        "external_predicted_class",
        element_wise=True,
    )
    @classmethod
    def valid_prediction(cls, value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, Integral)
            and int(value) >= 0
        )

    class Config(pa.DataFrameModel.Config):
        strict: bool | Literal["filter"] = True
        unique_column_names: bool = True


class EvaluationSchema(BatchEvaluationSchema):
    """
    Evaluation rows that also identify each episode's instance.

    `split_index` is the instance's position in the evaluated split and
    `generation_index` its position in the generated dataset (ADR 0004).
    """

    generation_index: Series[int] = pa.Field(ge=0)
    split_index: Series[int] = pa.Field(ge=0)

    @pa.dataframe_check
    @classmethod
    def one_instance_per_episode(cls, frame: pd.DataFrame) -> bool:
        per_episode = frame.groupby("episode_id")[
            ["generation_index", "split_index"]
        ].nunique()
        return bool((per_episode <= 1).all().all())


class PreProvenanceSavedEvaluationSchema(EvaluationSchema):
    """
    Evaluation rows as the evaluator saved them before ADR 0002.

    Such tables have no identity columns beyond `eval_seed` and
    `eval_hard_budget`; the transform step still accepts them.
    """

    eval_seed: Series[Any] = pa.Field(nullable=True)
    eval_hard_budget: Series[Any] = pa.Field(nullable=True)

    @pa.check("eval_seed", element_wise=True)
    @classmethod
    def valid_seed(cls, value: object) -> bool:
        return not isinstance(value, bool) and isinstance(value, Integral)

    @pa.check("eval_hard_budget", element_wise=True)
    @classmethod
    def valid_budget(cls, value: object) -> bool:
        return (
            not isinstance(value, bool)
            and isinstance(value, Real)
            and float(value) >= 0
        )


# The identity columns of a saved evaluation table, with their pandas dtypes
IDENTITY_DTYPES = {
    "afa_method": "string",
    "dataset": "string",
    "dataset_realization_index": "UInt64",
    "eval_split": "string",
    "initializer": "string",
    "train_seed": "UInt64",
    "train_hard_budget": "Float64",
    "train_soft_budget_param": "Float64",
    "eval_seed": "UInt64",
    "eval_hard_budget": "Float64",
    "eval_soft_budget_param": "Float64",
}


def identity_column(name: str, value: object, index: pd.Index) -> pd.Series:
    """Return identity column `name` holding `value`, null for `None`."""
    # Built as object first so that None becomes the nullable dtype's NA
    return pd.Series(value, index=index, dtype=object).astype(
        IDENTITY_DTYPES[name]
    )


class SavedEvaluationSchema(EvaluationSchema):
    """
    Evaluation rows with the identity columns the evaluator adds (ADR 0002).

    Identity columns are constant per table and use pandas nullable dtypes,
    which Parquet round-trips. A null value is unknown: its source bundle
    predates provenance, or the setting (a budget) was not given.
    """

    afa_method: Series[pd.StringDtype] = pa.Field(nullable=True)
    dataset: Series[pd.StringDtype] = pa.Field(nullable=True)
    dataset_realization_index: Series[pd.UInt64Dtype] = pa.Field(nullable=True)
    eval_split: Series[pd.StringDtype] = pa.Field(
        nullable=True, isin=["train", "val", "test"]
    )
    initializer: Series[pd.StringDtype] = pa.Field()
    train_seed: Series[pd.UInt64Dtype] = pa.Field(nullable=True)
    train_hard_budget: Series[pd.Float64Dtype] = pa.Field(nullable=True, ge=0)
    train_soft_budget_param: Series[pd.Float64Dtype] = pa.Field(nullable=True)
    eval_seed: Series[pd.UInt64Dtype] = pa.Field()
    eval_hard_budget: Series[pd.Float64Dtype] = pa.Field(nullable=True, ge=0)
    eval_soft_budget_param: Series[pd.Float64Dtype] = pa.Field(nullable=True)

    @pa.dataframe_check
    @classmethod
    def constant_identity(cls, frame: pd.DataFrame) -> bool:
        return len(frame[list(IDENTITY_DTYPES)].drop_duplicates()) <= 1
