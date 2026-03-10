import pandas as pd
import torch

from afabench.common.datasets.datasets import CubeNMARDataset
from afabench.eval.cube_nm_ar import (
    augment_cube_nm_ar_eval_df,
    summarize_cube_nm_ar_episodes,
)


def _make_stop_episode_rows(
    sample_idx: int,
    actions: list[int],
    *,
    predicted_class: int,
    true_class: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    prev: list[int] = []
    for action in actions:
        rows.append(
            {
                "idx": sample_idx,
                "prev_selections_performed": prev.copy(),
                "action_performed": action,
                "external_predicted_class": predicted_class,
                "true_class": true_class,
                "accumulated_cost": float(len(prev)),
                "forced_stop": False,
            }
        )
        if action > 0:
            prev.append(action - 1)
    return rows


def test_cube_nm_ar_episode_summary_tracks_unsafe_stops() -> None:
    dataset = CubeNMARDataset(n_samples=3, seed=0)
    features, _labels = dataset.get_all_data()
    features.zero_()

    # Risky, unblocked context 3.
    features[0, 3] = 1.0
    # Risky, blocked context 4.
    features[1, 4] = 1.0
    # Risky, unblocked context 2.
    features[2, 2] = 1.0

    admin_start = dataset.n_contexts + dataset.n_hint_features
    features[1, admin_start + 1] = 1.0
    dataset.features = features

    rescue_action = 2 + dataset.n_contexts * dataset.block_size
    eval_df = pd.DataFrame(
        _make_stop_episode_rows(
            0,
            [1, 14, rescue_action, 0],
            predicted_class=3,
            true_class=3,
        )
        + _make_stop_episode_rows(
            1,
            [1, 0],
            predicted_class=1,
            true_class=2,
        )
        + _make_stop_episode_rows(
            2,
            [1, 10, 0],
            predicted_class=4,
            true_class=5,
        )
    )

    augmented = augment_cube_nm_ar_eval_df(eval_df, dataset)
    summary = summarize_cube_nm_ar_episodes(augmented)

    assert summary is not None
    summary = summary.sort_values("idx").reset_index(drop=True)

    assert torch.equal(
        torch.tensor(summary["relevant_block_acquired"].tolist()),
        torch.tensor([True, False, True]),
    )
    assert torch.equal(
        torch.tensor(summary["rescue_acquired"].tolist()),
        torch.tensor([True, False, False]),
    )
    assert torch.equal(
        torch.tensor(summary["unsafe_stop"].tolist()),
        torch.tensor([False, True, True]),
    )
    assert torch.equal(
        torch.tensor(summary["avoidable_unsafe_stop"].tolist()),
        torch.tensor([False, False, True]),
    )
