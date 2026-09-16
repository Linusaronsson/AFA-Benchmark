from pathlib import Path

import pandas as pd
import pytest

from scripts.plotting.plot_state_conditioning import (
    FAMILY_METHODS,
    SOURCES,
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


def test_state_conditioning_collects_all_paired_cells(tmp_path: Path) -> None:
    records = []
    for dataset in SOURCES["induced"]:
        for methods in FAMILY_METHODS.values():
            for method in methods:
                for instance in range(5):
                    for strategy in (
                        "complete",
                        "restricted",
                        "pvae_label_conditioned",
                    ):
                        rates = (
                            (0.0,)
                            if strategy == "complete"
                            else (0.3, 0.5, 0.7)
                        )
                        for rate in rates:
                            row = _state_row(method, strategy, 0.6)
                            row.update(
                                dataset=dataset, instance=instance, p=rate
                            )
                            records.append(row)
    root = tmp_path / "induced"
    root.mkdir()
    pd.DataFrame(records).to_csv(root / "instance_metrics.csv", index=False)
    frame = collect(tmp_path)

    assert len(frame) == 72
    assert set(frame.groupby("family").size()) == {24}
    assert (frame[["n_restricted", "n_restored"]] == 5).all().all()
    assert (frame[["restricted", "restored"]] == 0).all().all()
