from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.analyze_missing_data_mechanisms import (
    _evaluation_indices,
    _positioned_final_rows,
    _sampled_indices,
    _trace_columns,
    cube_nm_episode_metrics,
    generator_quality_pairs,
    stepwise_effects,
)
from scripts.plotting.plot_missing_data_mechanisms import (
    plot_generator_quality,
    plot_path_fidelity,
    plot_stepwise,
)


def test_positioned_final_rows_recovers_batch_local_indices() -> None:
    frame = pd.DataFrame(
        {
            "idx": [0, 1, 0, 0],
            "prev_selections_performed": [[], [], [4], []],
            "action_performed": [4, 0, 0, 0],
        }
    )

    final = _positioned_final_rows(frame)

    assert final["_position"].tolist() == [0, 1, 2]
    assert final["idx"].tolist() == [0, 1, 0]


def test_trace_comparison_uses_global_episode_position() -> None:
    frame = pd.DataFrame(
        {
            "idx": [0, 1, 0, 0],
            "prev_selections_performed": [[], [], [4], []],
            "action_performed": [4, 0, 0, 0],
            "external_predicted_class": [0, 1, 1, 0],
            "true_class": [0, 1, 1, 0],
            "accumulated_cost": [0.0, 0.0, 1.0, 0.0],
            "forced_stop": [False, False, True, False],
        }
    )

    trace = _trace_columns(_positioned_final_rows(frame))

    assert trace["_position"].tolist() == [0, 1, 2]
    assert trace["prev_selections_performed"].tolist() == [(4,), (), ()]


def test_evaluation_indices_recovers_seeded_subset_order() -> None:
    labels = np.arange(20)
    expected = _sampled_indices(len(labels), 12, seed=7)

    actual = _evaluation_indices(labels, labels[expected], seed=7)

    np.testing.assert_array_equal(actual, expected)


def test_cube_nm_path_fidelity_uses_context_then_branch() -> None:
    metrics = cube_nm_episode_metrics(
        np.asarray([0, 1, 2, 1]),
        [
            [0, 1, 2],
            [0, 11, 12],
            [3, 0, 21],
            [0],
        ],
    )

    assert metrics["context_first_rate"] == pytest.approx(0.75)
    assert metrics["n_correct_next_eligible"] == 2
    assert metrics["correct_next_rate"] == pytest.approx(1.0)
    assert metrics["correct_block_allocation"] == pytest.approx(1.0)


def _instance_metrics() -> pd.DataFrame:
    rows = []
    for strategy, score in [
        ("restricted", 0.60),
        ("pvae_label_conditioned", 0.66),
        ("pvae_stepwise", 0.69),
        ("pvae_oracle", 0.72),
    ]:
        rows.append(
            {
                "dataset": "cube_nm",
                "method": "aaco",
                "mechanism": "mcar",
                "p": 0.7,
                "strategy": strategy,
                "instance": 0,
                "eval_hard_budget": 14.0,
                "accuracy": score,
                "f_score": 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_stepwise_effects_are_exactly_paired() -> None:
    effects = stepwise_effects(_instance_metrics())

    assert effects.loc[0, "one_shot_gain"] == pytest.approx(0.06)
    assert effects.loc[0, "stepwise_gain"] == pytest.approx(0.09)
    assert effects.loc[0, "stepwise_minus_one_shot"] == pytest.approx(0.03)


def test_generator_quality_pairs_exact_cells() -> None:
    restoration = pd.DataFrame(
        [
            {
                "dataset": "cube_nm",
                "mechanism": "mcar",
                "p": 0.7,
                "instance": 0,
                "strategy": "pvae_label_conditioned",
                "split": "val",
                "imputation_rmse": 0.5,
            },
            {
                "dataset": "cube_nm",
                "mechanism": "mcar",
                "p": 0.7,
                "instance": 0,
                "strategy": "pvae_oracle",
                "split": "val",
                "imputation_rmse": 0.3,
            },
        ]
    )

    paired = generator_quality_pairs(_instance_metrics(), restoration)

    assert paired.loc[0, "rmse_improvement"] == pytest.approx(0.2)
    assert paired.loc[0, "score_improvement"] == pytest.approx(0.06)


def test_mechanism_figures_render(tmp_path: Path) -> None:
    path_rows = []
    for method in ("aaco", "ol_without_mask"):
        for strategy, mechanism, probability in [
            ("complete", "none", 0.0),
            ("restricted", "mcar", 0.7),
            ("pvae_label_conditioned", "mcar", 0.7),
            ("pvae_stepwise", "mcar", 0.7),
        ]:
            path_rows.append(
                {
                    "method": method,
                    "strategy": strategy,
                    "mechanism": mechanism,
                    "p": probability,
                    "context_first_rate": 0.8,
                    "correct_next_rate": 0.7,
                    "correct_block_allocation": 0.6,
                }
            )
    plot_path_fidelity(pd.DataFrame(path_rows), tmp_path)
    plot_generator_quality(
        pd.DataFrame(
            {
                "method": ["aaco", "ol_without_mask"],
                "rmse_improvement": [0.2, 0.1],
                "score_improvement": [0.05, 0.02],
            }
        ),
        tmp_path,
    )
    plot_stepwise(
        pd.DataFrame(
            {
                "method": ["aaco", "ol_without_mask"],
                "dataset": ["cube_nm", "cube_nm"],
                "stepwise_minus_one_shot": [0.03, 0.01],
            }
        ),
        tmp_path,
    )

    assert (tmp_path / "cube_nm_path_fidelity.pdf").exists()
    assert (tmp_path / "generator_quality_vs_score.svg").exists()
    assert (tmp_path / "stepwise_vs_one_shot.pdf").exists()
