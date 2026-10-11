"""On-demand selection histories for a single evaluation artifact."""

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
