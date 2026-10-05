"""Contracts for evaluation rows and metadata-enriched Parquet results."""

from numbers import Integral, Real
from typing import Any, Literal

import pandera.pandas as pa
from pandera.typing import Series


class EvaluationSchema(pa.DataFrameModel):
    """One row per sample and acquisition timestep (not a unique sample)."""

    prev_selections_performed: Series[list[int]] = pa.Field()
    action_performed: Series[int] = pa.Field(ge=0)
    # Predictions use int64 when supplied and object/None otherwise. Avoid
    # coercion so existing callers and Parquet artifacts keep their dtypes.
    builtin_predicted_class: Series[Any] = pa.Field(nullable=True)
    external_predicted_class: Series[Any] = pa.Field(nullable=True)
    true_class: Series[int] = pa.Field(ge=0)
    accumulated_cost: Series[float] = pa.Field(ge=0)
    idx: Series[int] = pa.Field(ge=0)
    forced_stop: Series[bool] = pa.Field()

    @pa.check("prev_selections_performed", element_wise=True)
    @classmethod
    def nonnegative_selections(cls, value: list[int]) -> bool:
        return all(
            not isinstance(selection, bool) and selection >= 0
            for selection in value
        )

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


class SavedEvaluationSchema(EvaluationSchema):
    """Evaluation rows with the run metadata added by the evaluator."""

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
