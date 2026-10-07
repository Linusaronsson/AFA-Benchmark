from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch
import wandb

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.core.bundle_system.bundle import save_bundle
from afabench.core.provenance import DatasetIdentityMismatchError, Split
from afabench.datasets.datasets import CubeDataset
from afabench.fit.contract import PretrainingContract, TrainingContract
from afabench.fit.metric_logger import NullMetricLogger
from afabench.fit.run import fit_run
from afabench.testing.provenance import placeholder_provenance


@dataclass(frozen=True)
class _MethodTrainConfig(TrainingContract):
    learning_rate: float


def _save_dataset_bundle(
    path: Path, *, split: Split, dataset_realization_index: int
) -> None:
    save_bundle(
        CubeDataset(n_samples=6, seed=dataset_realization_index),
        path,
        metadata={},
        provenance=placeholder_provenance(
            dataset_key="cube",
            dataset_realization_index=dataset_realization_index,
            split=split,
        ),
    )


@pytest.fixture(autouse=True)
def dataset_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the contracts' relative dataset bundles; fit_run reads them."""
    monkeypatch.chdir(tmp_path)
    _save_dataset_bundle(
        Path("train.bundle"), split="train", dataset_realization_index=2
    )
    _save_dataset_bundle(
        Path("val.bundle"), split="val", dataset_realization_index=2
    )
    return tmp_path


def _method_config(*, seed: int, use_wandb: bool) -> _MethodTrainConfig:
    return _MethodTrainConfig(
        train_dataset_bundle_path="train.bundle",
        val_dataset_bundle_path="val.bundle",
        classifier_bundle_path="classifier.bundle",
        save_path="method.bundle",
        initializer=InitializerConfig(
            class_name="RandomInitializer",
            kwargs={"num_initial_features": 0},
        ),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        method_name="my_method",
        hard_budget=3,
        soft_budget_param=None,
        device="cpu",
        seed=seed,
        use_wandb=use_wandb,
        learning_rate=0.1,
    )


class _FakeWandbRun:
    name = "fake-run"
    id = "fake-id"
    url = "https://wandb.invalid/fake-run"

    def __init__(self, init_kwargs: Mapping[str, object]) -> None:
        self.init_kwargs = init_kwargs
        self.logged: list[dict[str, object]] = []
        self.finished = False

    def log(self, data: dict[str, object]) -> None:
        self.logged.append(data)

    def finish(self) -> None:
        self.finished = True


@pytest.fixture
def cuda_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Report CUDA as available and record the cleanup calls made to it."""
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *_: calls.append("synchronize")
    )
    monkeypatch.setattr(
        torch.cuda, "empty_cache", lambda: calls.append("empty_cache")
    )
    return calls


@pytest.fixture
def fake_wandb_runs(monkeypatch: pytest.MonkeyPatch) -> list[_FakeWandbRun]:
    runs: list[_FakeWandbRun] = []

    def fake_init(**kwargs: object) -> _FakeWandbRun:
        run = _FakeWandbRun(kwargs)
        runs.append(run)
        return run

    monkeypatch.delenv("WANDB_GROUP", raising=False)
    monkeypatch.setattr(wandb, "init", fake_init)
    return runs


def test_fit_run_seeds_from_the_contract() -> None:
    config = _method_config(seed=11, use_wandb=False)

    with fit_run(config, tags=["m"], config=config):
        first = torch.rand(3)
    with fit_run(config, tags=["m"], config=config):
        second = torch.rand(3)
    other_config = replace(config, seed=12)
    with fit_run(other_config, tags=["m"], config=other_config):
        other = torch.rand(3)

    assert torch.equal(first, second)
    assert not torch.equal(first, other)


def test_fit_run_on_cpu_does_not_touch_cuda(
    cuda_calls: list[str],
) -> None:
    config = _method_config(seed=0, use_wandb=False)

    with fit_run(config, tags=["m"], config=config):
        pass

    assert cuda_calls == []


def test_fit_run_on_cuda_releases_cuda_memory_afterwards(
    cuda_calls: list[str],
) -> None:
    config = replace(_method_config(seed=0, use_wandb=False), device="cuda")

    with fit_run(config, tags=["m"], config=config):
        assert cuda_calls == []

    assert cuda_calls == ["empty_cache", "synchronize"]


def test_fit_run_without_wandb_logs_nowhere(
    fake_wandb_runs: list[_FakeWandbRun],
) -> None:
    config = _method_config(seed=0, use_wandb=False)

    with fit_run(config, tags=["m"], config=config) as metric_logger:
        metric_logger.log({"loss": 1.0})

    assert isinstance(metric_logger, NullMetricLogger)
    assert fake_wandb_runs == []


@pytest.mark.parametrize(
    ("contract_class", "stage"),
    [(PretrainingContract, "pretraining"), (TrainingContract, "training")],
)
def test_fit_run_with_wandb_sets_job_type_from_contract_stage(
    fake_wandb_runs: list[_FakeWandbRun],
    contract_class: type[PretrainingContract | TrainingContract],
    stage: str,
) -> None:
    kwargs = (
        {
            "method_name": "my_method",
            "hard_budget": 3,
            "soft_budget_param": None,
        }
        if contract_class is TrainingContract
        else {}
    )
    contract = contract_class(
        train_dataset_bundle_path="train.bundle",
        val_dataset_bundle_path="val.bundle",
        classifier_bundle_path="classifier.bundle",
        save_path="method.bundle",
        initializer=InitializerConfig(
            class_name="RandomInitializer",
            kwargs={"num_initial_features": 0},
        ),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        device="cpu",
        seed=0,
        use_wandb=True,
        **kwargs,  # pyright: ignore[reportArgumentType]
    )

    with fit_run(contract, tags=["my_method"], config=contract):
        pass

    [run] = fake_wandb_runs
    assert run.init_kwargs["job_type"] == stage
    assert run.init_kwargs["tags"] == ["my_method"]


def test_fit_run_with_wandb_logs_the_flat_method_config(
    fake_wandb_runs: list[_FakeWandbRun],
) -> None:
    config = _method_config(seed=5, use_wandb=True)

    with fit_run(config, tags=["m"], config=config):
        pass

    [run] = fake_wandb_runs
    logged_config = run.init_kwargs["config"]
    assert isinstance(logged_config, dict)
    assert logged_config["seed"] == 5
    assert logged_config["learning_rate"] == 0.1
    assert logged_config["unmasker"] == {
        "class_name": "DirectUnmasker",
        "kwargs": {},
    }


def test_fit_run_with_wandb_forwards_metrics_and_finishes(
    fake_wandb_runs: list[_FakeWandbRun],
) -> None:
    config = _method_config(seed=0, use_wandb=True)

    with fit_run(config, tags=["m"], config=config) as metric_logger:
        metric_logger.log({"loss": 0.5})
        [run] = fake_wandb_runs
        assert not run.finished

    assert run.logged == [{"loss": 0.5}]
    assert run.finished


def test_fit_run_finishes_wandb_when_training_fails(
    fake_wandb_runs: list[_FakeWandbRun],
) -> None:
    config = _method_config(seed=0, use_wandb=True)

    def train() -> None:
        with fit_run(config, tags=["m"], config=config):
            msg = "diverged"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="diverged"):
        train()

    [run] = fake_wandb_runs
    assert run.finished


def test_fit_run_rejects_datasets_of_different_realizations_before_training(
    dataset_bundles: Path,
) -> None:
    _save_dataset_bundle(
        dataset_bundles / "val.bundle",
        split="val",
        dataset_realization_index=3,
    )
    config = _method_config(seed=0, use_wandb=False)
    trained = False

    def train() -> None:
        nonlocal trained
        with fit_run(config, tags=["m"], config=config):
            trained = True

    with pytest.raises(DatasetIdentityMismatchError, match=r"2.*3"):
        train()
    assert not trained
