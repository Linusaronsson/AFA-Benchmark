from pathlib import Path

import pandas as pd
import pytest

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.evaluation.config import EvalConfig
from scripts.eval.eval_afa_method import AFAEvaluator


class WandbInitSpy:
    def __init__(self) -> None:
        self.config: dict[str, object] | None = None

    def __call__(
        self,
        *,
        job_type: str,
        config: dict[str, object],
        dir: str,  # noqa: A002
    ) -> object:
        assert job_type == "evaluation"
        assert dir == "extra/logs/wandb"
        self.config = config
        return WandbRunStub()


class WandbRunStub:
    name = "test-run"
    id = "test-id"
    url = "https://wandb.example/test-run"


def test_init_wandb_accepts_dataclass_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_spy = WandbInitSpy()
    monkeypatch.setattr("scripts.eval.eval_afa_method.wandb.init", init_spy)

    evaluator = AFAEvaluator(eval_config(use_wandb=True))

    evaluator._init_wandb()  # noqa: SLF001

    assert init_spy.config == {
        "method_bundle_path": "method.bundle",
        "unmasker": {"class_name": "DirectUnmasker", "kwargs": {}},
        "initializer": {"class_name": "ColdInitializer", "kwargs": {}},
        "dataset_bundle_path": "dataset.bundle",
        "save_path": "eval.csv",
        "classifier_bundle_path": None,
        "seed": 1,
        "device": "cpu",
        "eval_only_n_samples": None,
        "batch_size": 8,
        "hard_budget": None,
        "soft_budget_param": None,
        "use_wandb": True,
        "smoke_test": False,
    }


def test_smoke_test_override_uses_two_batches() -> None:
    cfg = eval_config(use_wandb=False, smoke_test=True)
    evaluator = AFAEvaluator(cfg)

    evaluator._smoke_test_override()  # noqa: SLF001

    assert cfg.eval_only_n_samples == 4
    assert cfg.batch_size == 2


def test_save_writes_parquet(tmp_path: Path) -> None:
    save_path = tmp_path / "eval.parquet"
    cfg = eval_config(use_wandb=False)
    cfg.save_path = str(save_path)
    evaluator = AFAEvaluator(cfg)
    evaluator._df_eval = pd.DataFrame(  # noqa: SLF001
        {
            "prev_selections_performed": [[0], []],
            "action_performed": [1, 0],
            "builtin_predicted_class": [None, None],
            "external_predicted_class": [1, 0],
            "true_class": [1, 0],
            "accumulated_cost": [1.0, 0.0],
            "idx": [0, 1],
            "forced_stop": [False, False],
            "eval_seed": [1, 1],
            "eval_hard_budget": [None, None],
        }
    )

    evaluator._save()  # noqa: SLF001

    saved = pd.read_parquet(save_path)
    saved_records = normalize_selection_records(saved)
    expected_records = normalize_selection_records(evaluator._df_eval)  # noqa: SLF001
    assert saved_records == expected_records


def normalize_selection_records(
    df: pd.DataFrame,
) -> list[dict[str, object]]:
    records = df.to_dict("records")
    for record in records:
        selections = record["prev_selections_performed"]
        record["prev_selections_performed"] = list(selections)
    return records


def eval_config(*, use_wandb: bool, smoke_test: bool = False) -> EvalConfig:
    return EvalConfig(
        method_bundle_path="method.bundle",
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        initializer=InitializerConfig(class_name="ColdInitializer", kwargs={}),
        dataset_bundle_path="dataset.bundle",
        save_path="eval.csv",
        classifier_bundle_path=None,
        seed=1,
        device="cpu",
        eval_only_n_samples=None,
        batch_size=8,
        use_wandb=use_wandb,
        smoke_test=smoke_test,
    )
