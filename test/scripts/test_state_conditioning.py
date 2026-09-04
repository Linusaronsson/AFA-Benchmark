from pathlib import Path

import pandas as pd
import pytest

from scripts.plotting.plot_state_conditioning import (
    _paired_state_difference,
    collect,
)


def _state_row(
    method: str,
    strategy: str,
    score: float,
) -> dict[str, object]:
    complete = strategy == "complete"
    return {
        "dataset": "cube",
        "method": method,
        "mechanism": "none" if complete else "mcar",
        "p": 0.0 if complete else 0.5,
        "strategy": strategy,
        "instance": 0,
        "train_hard_budget": 10.0,
        "eval_hard_budget": 10.0,
        "accuracy": score,
        "f_score": score,
    }


def test_state_difference_is_paired_and_complete_adjusted() -> None:
    frame = pd.DataFrame(
        [
            _state_row("jafa", "complete", 0.5),
            _state_row("jafa_full_state", "complete", 0.6),
            _state_row("jafa", "restricted", 0.4),
            _state_row("jafa_full_state", "restricted", 0.7),
            _state_row("jafa", "pvae_label_conditioned", 0.55),
            _state_row("jafa_full_state", "pvae_label_conditioned", 0.60),
        ]
    )

    paired = _paired_state_difference(
        frame, "cube", "jafa", "jafa_full_state", "mcar"
    )

    adjusted = paired.set_index("strategy")["adjusted_difference"]
    assert adjusted["restricted"] == pytest.approx(0.2)
    assert adjusted["pvae_label_conditioned"] == pytest.approx(-0.05)


def test_production_state_conditioning_has_full_coverage() -> None:
    root = Path("extra/output/missing_data/summary/val")

    frame = collect(root)

    assert len(frame) == 72
    assert set(frame.groupby("family").size()) == {24}
    assert (frame[["n_restricted", "n_restored"]] == 5).all().all()
    means = frame.groupby("family")[["restricted", "restored"]].mean()
    assert means.loc["JAFA", "restricted"] == pytest.approx(-0.0333, abs=5e-5)
    assert means.loc["JAFA", "restored"] == pytest.approx(0.0012, abs=5e-5)
    assert means.loc["OL", "restricted"] == pytest.approx(-0.0677, abs=5e-5)
    assert means.loc["OL", "restored"] == pytest.approx(-0.0121, abs=5e-5)
    assert means.loc["ODIN", "restricted"] == pytest.approx(-0.0034, abs=5e-5)
    assert means.loc["ODIN", "restored"] == pytest.approx(-0.0073, abs=5e-5)
