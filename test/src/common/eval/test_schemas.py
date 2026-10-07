"""Compact evaluation contracts validate without coercing dataframe dtypes."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pandera.errors import SchemaError, SchemaErrors
from pandera.typing import DataFrame

from afabench.evaluation.schemas import (
    EvaluationSchema,
    SavedEvaluationSchema,
)


@pytest.fixture
def evaluation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": [0, 0],
            "generation_index": [5, 5],
            "split_index": [2, 2],
            "step": [0, 1],
            "action_performed": [1, 0],
            "builtin_predicted_class": [None, None],
            "external_predicted_class": [1, 0],
            "true_class": [1, 1],
            "accumulated_cost": [1.0, 1.0],
            "forced_stop": [False, True],
        }
    )


def test_evaluation_schema_preserves_dtypes(
    evaluation_frame: pd.DataFrame,
) -> None:
    validated = DataFrame[EvaluationSchema](evaluation_frame)
    assert_frame_equal(validated, evaluation_frame)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("action_performed", -1),
        ("true_class", -1),
        ("episode_id", -1),
        ("generation_index", -1),
        ("split_index", -1),
        ("step", -1),
        ("accumulated_cost", -0.5),
        ("forced_stop", "false"),
        ("builtin_predicted_class", "1"),
        ("builtin_predicted_class", -1),
        ("step", "0"),
    ],
)
def test_evaluation_schema_rejects_invalid_values(
    evaluation_frame: pd.DataFrame, column: str, value: object
) -> None:
    evaluation_frame[column] = pd.Series([value, value])
    with pytest.raises(SchemaError):
        EvaluationSchema.validate(evaluation_frame)


@pytest.mark.parametrize("column", list(EvaluationSchema.to_schema().columns))
def test_evaluation_schema_requires_all_columns(
    evaluation_frame: pd.DataFrame, column: str
) -> None:
    with pytest.raises(SchemaError):
        EvaluationSchema.validate(evaluation_frame.drop(columns=column))


def test_evaluation_schema_rejects_incomplete_episodes(
    evaluation_frame: pd.DataFrame,
) -> None:
    with pytest.raises(SchemaError):
        EvaluationSchema.validate(evaluation_frame.iloc[1:])


@pytest.mark.parametrize("column", ["generation_index", "split_index"])
def test_evaluation_schema_rejects_episode_spanning_instances(
    evaluation_frame: pd.DataFrame, column: str
) -> None:
    evaluation_frame[column] = [0, 1]
    with pytest.raises(SchemaError):
        EvaluationSchema.validate(evaluation_frame)


def test_evaluation_schema_rejects_extra_columns(
    evaluation_frame: pd.DataFrame,
) -> None:
    with pytest.raises((SchemaError, SchemaErrors)):
        EvaluationSchema.validate(evaluation_frame.assign(unexpected=0))


@pytest.mark.parametrize("seed", [None, 42])
@pytest.mark.parametrize("budget", [None, 2, 2.5])
def test_saved_evaluation_schema_parquet_round_trip(
    evaluation_frame: pd.DataFrame,
    tmp_path: Path,
    seed: int | None,
    budget: float | None,
) -> None:
    frame = evaluation_frame.assign(eval_seed=seed, eval_hard_budget=budget)
    validated = DataFrame[SavedEvaluationSchema](frame)
    assert_frame_equal(validated, frame)
    path = tmp_path / "eval.parquet"
    validated.to_parquet(path, index=False)
    loaded = pd.read_parquet(path)
    SavedEvaluationSchema.validate(loaded)


@pytest.mark.parametrize(
    ("column", "value"),
    [("eval_seed", "42"), ("eval_hard_budget", -1)],
)
def test_saved_schema_rejects_invalid_metadata(
    evaluation_frame: pd.DataFrame, column: str, value: object
) -> None:
    frame = evaluation_frame.assign(eval_seed=None, eval_hard_budget=None)
    frame[column] = value
    with pytest.raises(SchemaError):
        SavedEvaluationSchema.validate(frame)
