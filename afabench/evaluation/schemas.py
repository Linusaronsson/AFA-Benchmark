"""Contracts for evaluation rows and metadata-enriched Parquet results."""

from numbers import Integral, Real
from typing import Any, Literal

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

from afabench.evaluation.history import validate_episode_log


class EvaluationSchema(pa.DataFrameModel):
    """One row per episode and zero-based time step."""

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
