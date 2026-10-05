"""Selection history is derived only from complete episode logs."""

from pathlib import Path

import pandas as pd
import pytest

from afabench.evaluation.history import (
    convert_legacy_episode_log,
    reconstruct_selection_history,
)


def test_converts_complete_original_order_legacy_log() -> None:
    legacy = pd.DataFrame(
        {
            "idx": [0, 1, 0, 0, 0, 0],
            "action_performed": [3, 0, 3, 0, 1, 0],
            "prev_selections_performed": [
                [],
                [],
                [2],
                [2, 2],
                [],
                [0],
            ],
            "accumulated_cost": [0.0] * 6,
        }
    )
    compact = convert_legacy_episode_log(legacy, original_order=True)
    assert compact["episode_id"].tolist() == [0, 1, 0, 0, 2, 2]
    assert compact["step"].tolist() == [0, 0, 1, 2, 0, 1]
    assert "idx" not in compact
    assert "prev_selections_performed" not in compact
    assert reconstruct_selection_history(compact).tolist() == [
        [],
        [],
        [2],
        [2, 2],
        [],
        [0],
    ]
    assert "idx" in legacy


@pytest.mark.parametrize(
    ("indices", "actions", "histories"),
    [
        ([0], [1], [[]]),
        ([0], [0], [[2]]),
        ([0, 0], [2, 0], [[], [3, 1]]),
        ([0, 0], [0, 1], [[0], []]),
        ([-1], [0], [[]]),
        ([True], [0], [[]]),
        ([0, 0], [2, 0], [[], [True]]),
        ([0, 0], [1, 0], [[], [0.0]]),
        ([0, 1, 0, 0, 1], [1, 1, 0, 0, 0], [[], [], [0], [], [0]]),
        ([0, 1, 0, 1], [0, 1, 0, 0], [[], [], [], [0]]),
    ],
)
def test_legacy_conversion_rejects_invalid_or_partial_logs(
    indices: list[int | bool],
    actions: list[int],
    histories: list[list[int | bool | float]],
) -> None:
    legacy = pd.DataFrame(
        {
            "idx": indices,
            "action_performed": actions,
            "prev_selections_performed": histories,
        }
    )
    with pytest.raises(ValueError, match="Legacy"):
        convert_legacy_episode_log(legacy, original_order=True)


def test_legacy_batches_can_start_with_immediate_stops() -> None:
    legacy = pd.DataFrame(
        {
            "idx": [0, 1, 1, 0, 1, 0],
            "action_performed": [0, 2, 0, 1, 0, 0],
            "prev_selections_performed": [[], [], [1], [], [], [0]],
        }
    )
    compact = convert_legacy_episode_log(legacy, original_order=True)
    assert compact["episode_id"].tolist() == [0, 1, 1, 2, 3, 2]
    assert compact["step"].tolist() == [0, 0, 1, 0, 0, 1]


@pytest.mark.parametrize("string_history", [False, True])
def test_converts_legacy_histories_read_from_parquet(
    tmp_path: Path, string_history: bool
) -> None:
    frame = pd.DataFrame(
        {
            "idx": [0, 0],
            "action_performed": [2, 0],
            "prev_selections_performed": (
                ["[]", "[1]"] if string_history else [[], [1]]
            ),
        }
    )
    path = tmp_path / "legacy.parquet"
    frame.to_parquet(path, index=False)
    compact = convert_legacy_episode_log(
        pd.read_parquet(path), original_order=True
    )
    assert reconstruct_selection_history(compact).tolist() == [[], [1]]


def test_legacy_conversion_requires_explicit_original_order() -> None:
    with pytest.raises(ValueError, match="original producer row order"):
        convert_legacy_episode_log(pd.DataFrame())


def test_reconstructs_ordered_histories_after_reordering() -> None:
    frame = pd.DataFrame(
        {
            "episode_id": [7, 2, 7, 7, 2, 7],
            "step": [3, 1, 1, 0, 0, 2],
            "action_performed": [0, 0, 3, 3, 1, 1],
        },
        index=[4, 4, 1, 8, 2, 9],
    )
    histories = reconstruct_selection_history(frame)
    assert histories.tolist() == [[2, 2, 0], [0], [2], [], [], [2, 2]]
    assert histories.index.equals(frame.index)
    assert list(frame.columns) == ["episode_id", "step", "action_performed"]


@pytest.mark.parametrize(
    ("steps", "actions"),
    [
        ([1, 2], [1, 0]),
        ([0, 2], [1, 0]),
        ([0, 0], [1, 0]),
        ([0, 1], [1, 2]),
        ([0, 1], [0, 1]),
        ([0, 1], [0, 0]),
        ([0, -1], [1, 0]),
        ([0, 1], [-1, 0]),
        ([0, 1.5], [1, 0]),
        ([0, 1], [True, False]),
    ],
)
def test_reconstruction_rejects_malformed_episodes(
    steps: list[int | float], actions: list[int | bool]
) -> None:
    frame = pd.DataFrame(
        {"episode_id": [0, 0], "step": steps, "action_performed": actions}
    )
    with pytest.raises(ValueError, match=r"Episode|nonnegative integers"):
        reconstruct_selection_history(frame)
