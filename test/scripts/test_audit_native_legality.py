from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.audit_native_legality import audit_native_evaluations


def _write_trace(
    root: Path,
    *,
    strategy: str = "restricted",
    legal: bool = True,
) -> None:
    experiment = (
        "method-dime+mechanism-native+p-0.0+"
        f"strategy-{strategy}+instance-0+"
        "train_hard_budget-5+eval_hard_budget-5"
    )
    path = root / "dataset-ckd" / experiment / "eval_data.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "action_performed": [1, 0],
            "source_idx": [7, 7],
            "selection_was_legal": [legal, True],
            "respected_native_availability": [True, True],
            "feature_availability_fraction": [0.8, 0.8],
            "selection_availability_fraction": [0.75, 0.75],
        }
    ).to_parquet(path, index=False)


def test_audit_native_evaluations_reports_verified_trace(
    tmp_path: Path,
) -> None:
    _write_trace(tmp_path)

    report = audit_native_evaluations(tmp_path)

    assert report["dataset"].tolist() == ["ckd"]
    assert report["n_source_samples"].tolist() == [1]
    assert report["n_acquisitions"].tolist() == [1]
    assert report["n_illegal_actions"].tolist() == [0]


def test_audit_native_evaluations_rejects_illegal_action(
    tmp_path: Path,
) -> None:
    _write_trace(tmp_path, legal=False)

    with pytest.raises(ValueError, match="1 illegal native actions"):
        audit_native_evaluations(tmp_path)


def test_audit_native_evaluations_rejects_oracle_strategy(
    tmp_path: Path,
) -> None:
    _write_trace(tmp_path, strategy="true_completion")

    with pytest.raises(ValueError, match="prohibited strategy"):
        audit_native_evaluations(tmp_path)
