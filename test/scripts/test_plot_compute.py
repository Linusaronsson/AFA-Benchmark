from pathlib import Path

import pandas as pd
import pytest

from afabench.plotting.methods import PRIMARY_METHODS
from scripts.plotting.plot_compute import attribute, paired_costs, panel_cells


def _environment(commit: str = "commit-a") -> dict[str, object]:
    return {
        "hardware_signature": f"signature-{commit}",
        "git_commit": commit,
        "device": "cuda",
        "cores": 72,
        "mem_mb": 103_680,
        "gpu_workers": 32,
        "mps": True,
        "architecture": "aarch64",
        "torch": "2.11.0+cu130",
        "torch_cuda": 13.0,
        "cuda_devices": '["NVIDIA GH200 120GB"]',
    }


def _record(
    rule: str,
    *,
    strategy: str,
    wall_seconds: float,
    method: str | None = None,
    commit: str = "commit-a",
) -> dict[str, object]:
    return {
        "namespace": "study",
        "rule": rule,
        "dataset": "cube",
        "mechanism": "mcar",
        "p": 0.5,
        "strategy": strategy,
        "instance": 0,
        "method": method,
        "pretrain_key": None,
        "wall_seconds": wall_seconds,
        "cpu_seconds": wall_seconds * 2,
        "peak_rss_mb": 100.0,
        **_environment(commit),
    }


def test_generator_cost_is_shared_by_all_actual_restored_consumers() -> None:
    rows = [
        _record(
            "train_method",
            strategy=strategy,
            wall_seconds=10.0,
            method=method,
        )
        for method in PRIMARY_METHODS
        for strategy in ("restricted", "pvae_label_conditioned")
    ]
    rows.extend(
        [
            _record(
                "pretrain_restoration_pvae_incomplete",
                strategy="pvae_label_conditioned",
                wall_seconds=90.0,
                commit="generator-commit",
            ),
            _record(
                "restore_view",
                strategy="pvae_label_conditioned",
                wall_seconds=18.0,
            ),
        ]
    )

    costs = attribute(pd.DataFrame(rows))
    restored = costs.loc[costs["arm"] == "generative"]

    assert len(restored) == len(PRIMARY_METHODS)
    assert set(restored["generator_share_wall_seconds"]) == {10.0}
    assert set(restored["restore_share_wall_seconds"]) == {2.0}
    assert set(restored["wall_seconds"]) == {22.0}
    assert set(restored["generator_git_commit"]) == {"generator-commit"}


def _write_scores(root: Path) -> None:
    path = root / "study"
    path.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "dataset": "cube",
                "method": "aaco",
                "mechanism": "mcar",
                "p": 0.5,
                "instance": 0,
                "strategy": "restricted",
                "accuracy": 0.7,
                "f_score": 0.6,
            },
            {
                "dataset": "cube",
                "method": "aaco",
                "mechanism": "mcar",
                "p": 0.5,
                "instance": 0,
                "strategy": "pvae_label_conditioned",
                "accuracy": 0.8,
                "f_score": 0.7,
            },
        ]
    ).to_csv(path / "instance_metrics.csv", index=False)


def _cost_arm(arm: str, commit: str) -> dict[str, object]:
    strategy = (
        "restricted" if arm == "restricted" else "pvae_label_conditioned"
    )
    return {
        "namespace": "study",
        "dataset": "cube",
        "mechanism": "mcar",
        "p": 0.5,
        "strategy": strategy,
        "instance": 0,
        "method": "aaco",
        "arm": arm,
        "wall_seconds": 10.0 if arm == "restricted" else 15.0,
        "cpu_seconds": 20.0,
        "peak_rss_mb": 100.0,
        "generator_git_commit": "generator" if arm == "generative" else "",
        "generator_hardware_signature": (
            "generator-signature" if arm == "generative" else ""
        ),
        **_environment(commit),
    }


def test_pairing_allows_different_commits_on_the_same_hardware(
    tmp_path: Path,
) -> None:
    _write_scores(tmp_path)
    costs = pd.DataFrame(
        [
            _cost_arm("restricted", "old-commit"),
            _cost_arm("generative", "new-commit"),
        ]
    )
    costs.loc[costs["arm"] == "generative", "gpu_workers"] = 43

    paired = paired_costs(costs, tmp_path)

    assert len(paired) == 1
    row = paired.iloc[0]
    assert row["git_commit_restricted"] == "old-commit"
    assert row["git_commit_generative"] == "new-commit"
    assert row["gpu_workers_restricted"] == 32
    assert row["gpu_workers_generative"] == 43
    assert row["wall_time_ratio"] == pytest.approx(1.5)
    assert row["restoration_gain"] == pytest.approx(0.1)


def test_pairing_rejects_a_real_hardware_mismatch(tmp_path: Path) -> None:
    _write_scores(tmp_path)
    restricted = _cost_arm("restricted", "old-commit")
    generative = _cost_arm("generative", "new-commit")
    generative["cuda_devices"] = '["Different GPU"]'

    paired = paired_costs(pd.DataFrame([restricted, generative]), tmp_path)

    assert paired.empty


PRODUCTION_PAIRED = Path(
    "extra/output/paper/experiments/results/compute.paired.csv"
)


@pytest.mark.skipif(
    not PRODUCTION_PAIRED.exists(),
    reason="paper compute artifact is not present in a clean checkout",
)
def test_production_compute_panel_has_72_groups_of_five() -> None:
    cells = panel_cells(pd.read_csv(PRODUCTION_PAIRED))

    assert len(cells) == 72
