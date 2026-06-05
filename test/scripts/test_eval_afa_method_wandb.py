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
