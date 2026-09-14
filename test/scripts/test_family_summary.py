"""State averaging preserves instance pairing and fails on incomplete cells."""

import pandas as pd
import pytest

from scripts.plotting.family_summary import (
    FAMILY_MEMBERS,
    average_states,
    summarize_families,
)


def _scores() -> pd.DataFrame:
    records = []
    for members in FAMILY_MEMBERS.values():
        for method in members:
            for instance in range(5):
                for strategy in (
                    "complete",
                    "restricted",
                    "pvae_label_conditioned",
                ):
                    score = 0.8 if strategy == "complete" else 0.6
                    if strategy == "restricted" and method == "jafa":
                        score = instance / 5
                    if (
                        strategy == "restricted"
                        and method == "jafa_full_state"
                    ):
                        score = 1 - instance / 5
                    records.append(
                        {
                            "dataset": "cube",
                            "method": method,
                            "instance": instance,
                            "train_hard_budget": 10,
                            "eval_hard_budget": 10,
                            "metric": "accuracy",
                            "strategy": strategy,
                            "score": score,
                            "mechanism": (
                                "none" if strategy == "complete" else "mcar"
                            ),
                            "p": 0 if strategy == "complete" else 0.7,
                        }
                    )
    return pd.DataFrame(records)


def test_correlated_states_are_averaged_before_bootstrap() -> None:
    instances = average_states(_scores())
    assert len(instances) == 30
    result = summarize_families(instances).set_index("method")
    jafa = result.loc["jafa"]
    assert jafa["direct_abs"] == pytest.approx(0.5)
    # The state scores vary strongly, but every paired mean is exactly 0.5.
    assert jafa["direct_abs_lo"] == pytest.approx(0.5)
    assert jafa["direct_abs_hi"] == pytest.approx(0.5)
    assert jafa["damage"] == pytest.approx(0.3)
    assert jafa["gain"] == pytest.approx(0.1)
    assert jafa["n_instances"] == 5
    assert jafa["n_variants"] == 2
    assert result.loc["dime", "n_variants"] == 1
    assert result.loc["dime", "direct_abs"] == pytest.approx(0.6)


@pytest.mark.parametrize("strategy", ["complete", "restricted"])
def test_incomplete_pairs_fail(strategy: str) -> None:
    frame = _scores()
    keep = ~(
        (frame["method"] == "jafa_full_state")
        & (frame["instance"] == 0)
        & (frame["strategy"] == strategy)
    )
    with pytest.raises(ValueError, match=r"missing|incomplete"):
        average_states(frame.loc[keep])


def test_duplicate_scientific_cells_fail() -> None:
    frame = _scores()
    with pytest.raises(ValueError, match="duplicate"):
        average_states(pd.concat([frame, frame.iloc[:1]], ignore_index=True))


def test_missing_instance_is_not_silently_reweighted() -> None:
    instances = average_states(_scores())
    with pytest.raises(ValueError, match="five matched"):
        summarize_families(instances.loc[instances["instance"] != 0])


def test_training_budgets_need_only_match_within_family() -> None:
    frame = _scores()
    frame.loc[frame["method"].isin(["dime", "gdfs"]), "train_hard_budget"] = 20
    assert len(average_states(frame)) == 30
    frame.loc[frame["method"] == "jafa_full_state", "train_hard_budget"] = 15
    with pytest.raises(ValueError, match="unpaired family training budgets"):
        average_states(frame)
