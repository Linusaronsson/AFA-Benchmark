"""On-demand selection histories for a single evaluation artifact."""

import pandas as pd

from afabench.evaluation.schemas import validate_episode_log


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
