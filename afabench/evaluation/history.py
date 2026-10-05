"""On-demand selection histories for a single evaluation artifact."""

import ast
from numbers import Integral

import pandas as pd


def _nonnegative_integer(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Integral)
        and int(value) >= 0
    )


def validate_episode_log(frame: pd.DataFrame) -> None:
    """
    Require complete episodes within one artifact, regardless of row order.

    Raises ValueError for invalid identity/action types, missing or duplicate
    steps, or anything other than a single terminal stop per episode.
    """
    for column in ("episode_id", "step", "action_performed"):
        if column not in frame or not frame.columns.is_unique:
            msg = f"Missing or duplicate episode-log column: {column}"
            raise ValueError(msg)
        if not all(_nonnegative_integer(value) for value in frame[column]):
            msg = f"{column} must contain nonnegative integers"
            raise ValueError(msg)
    for episode_id, episode in frame.sort_values("step").groupby("episode_id"):
        if episode["step"].tolist() != list(range(len(episode))):
            msg = f"Episode {episode_id} has missing or duplicate steps"
            raise ValueError(msg)
        actions = episode["action_performed"]
        if actions.iloc[-1] != 0 or (actions.iloc[:-1] == 0).any():
            msg = f"Episode {episode_id} must have exactly one terminal stop"
            raise ValueError(msg)


def convert_legacy_episode_log(
    frame: pd.DataFrame, *, original_order: bool = False
) -> pd.DataFrame:
    """
    Convert a complete legacy log without changing the source dataframe.

    The caller must attest that rows retain their original producer order;
    batch-local ``idx`` alone cannot recover episode boundaries after sorting.
    Stored histories must match every preceding action. Lists, Parquet arrays
    and legacy string lists are accepted. Ambiguous/partial logs are rejected.
    """
    if not original_order:
        msg = "Legacy conversion requires original producer row order"
        raise ValueError(msg)
    required = {"idx", "action_performed", "prev_selections_performed"}
    if (
        not required.issubset(frame.columns)
        or not frame.columns.is_unique
        or {"episode_id", "step"}.intersection(frame.columns)
    ):
        msg = "Expected an unambiguous legacy episode log"
        raise ValueError(msg)
    active: dict[int, tuple[int, list[int]]] = {}
    episode_ids: list[int] = []
    steps: list[int] = []
    next_episode_id = 0
    for idx, action, stored in zip(
        frame["idx"],
        frame["action_performed"],
        frame["prev_selections_performed"],
        strict=True,
    ):
        if not _nonnegative_integer(idx) or not _nonnegative_integer(action):
            msg = "Legacy indices and actions must be nonnegative integers"
            raise ValueError(msg)
        if idx not in active:
            active[idx] = (next_episode_id, [])
            next_episode_id += 1
        episode_id, history = active[idx]
        stored_history = (
            ast.literal_eval(stored) if isinstance(stored, str) else stored
        )
        try:
            stored_selections = list(stored_history)
        except TypeError as exc:
            msg = "Legacy history must be a sequence of selections"
            raise ValueError(msg) from exc
        if (
            not all(_nonnegative_integer(value) for value in stored_selections)
            or stored_selections != history
        ):
            msg = (
                "Legacy history does not match the complete ordered action log"
            )
            raise ValueError(msg)
        episode_ids.append(episode_id)
        steps.append(len(history))
        if action == 0:
            del active[idx]
        else:
            history.append(int(action) - 1)
    if active:
        msg = "Legacy log contains unterminated episodes"
        raise ValueError(msg)
    compact = frame.drop(columns=["idx", "prev_selections_performed"]).copy()
    compact["episode_id"] = pd.Series(
        episode_ids, index=frame.index, dtype="int64"
    )
    compact["step"] = pd.Series(steps, index=frame.index, dtype="int64")
    validate_episode_log(compact)
    return compact


def reconstruct_selection_history(frame: pd.DataFrame) -> pd.Series:
    """
    Return pre-action histories aligned to input row positions and index.

    Episode IDs are artifact-local: do not combine independently produced logs
    before calling this function. Stop and initial observations are excluded;
    repeated selections are retained. Materializing all prefixes can require
    quadratic memory, so callers needing only lengths should use ``step``.
    """
    validate_episode_log(frame)
    events = pd.DataFrame(
        frame[["episode_id", "step", "action_performed"]]
    ).copy()
    events["position"] = range(len(events))
    histories: list[list[int]] = [[] for _ in range(len(events))]
    for _, episode in events.sort_values("step").groupby("episode_id"):
        history: list[int] = []
        for position, action in zip(
            episode["position"], episode["action_performed"], strict=True
        ):
            histories[position] = history.copy()
            if action > 0:
                history.append(int(action) - 1)
    return pd.Series(
        histories,
        index=frame.index,
        name="prev_selections_performed",
        dtype=object,
    )
