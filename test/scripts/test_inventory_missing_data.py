from pathlib import Path

import pandas as pd

from scripts.analysis.inventory_missing_data import (
    build_coverage,
    build_inventory,
)


def _write_result(
    root: Path,
    namespace: str,
    *,
    instance: int,
    score: float,
) -> None:
    path = root / "val" / namespace / "instance_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "dataset": "cube",
                "method": "aaco",
                "mechanism": "mcar",
                "p": 0.5,
                "strategy": "restricted",
                "instance": instance,
                "train_hard_budget": 10,
                "eval_hard_budget": 10,
                "n_samples": 100,
                "accuracy": score,
                "f_score": score - 0.1,
            }
        ]
    ).to_csv(
        path,
        index=False,
        mode="a" if path.exists() else "w",
        header=not path.exists(),
    )


def test_inventory_preserves_duplicate_sources(tmp_path: Path) -> None:
    _write_result(tmp_path, "one", instance=0, score=0.7)
    _write_result(tmp_path, "two", instance=0, score=0.8)

    inventory = build_inventory(tmp_path)

    assert len(inventory) == 2
    assert set(inventory["namespace"]) == {"one", "two"}
    assert set(inventory["training_condition"]) == {"direct"}
    assert set(inventory["primary_metric"]) == {"accuracy"}
    assert set(inventory["duplicate_sources"]) == {2}
    assert inventory["ready_for_analysis"].tolist() == [False, False]


def test_coverage_counts_complete_instances(tmp_path: Path) -> None:
    for instance in range(5):
        _write_result(
            tmp_path,
            "complete",
            instance=instance,
            score=0.7 + instance / 100,
        )

    coverage = build_coverage(build_inventory(tmp_path))

    assert len(coverage) == 1
    assert coverage.loc[0, "n_instances"] == 5
    assert coverage.loc[0, "instances"] == "0,1,2,3,4"
    assert coverage["complete_five_instances"].tolist() == [True]
