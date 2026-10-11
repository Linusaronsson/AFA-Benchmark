from dataclasses import dataclass

import torch

from afabench.components.initializers.config import InitializerConfig
from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.fit.config import (
    SupervisedLearningConfig,
    SupervisedPretrainingConfig,
)
from afabench.fit.contract import PretrainingContract
from afabench.fit.smoke_test import (
    dataset_subset,
    eval_settings,
    limit,
    limit_supervised_learning,
    training_batch_size,
    training_subset,
)


class _SmokeDataset:
    feature_costs: torch.Tensor | None = None

    def __init__(self, values: list[int]) -> None:
        self.values = values

    @property
    def feature_shape(self) -> torch.Size:
        return torch.Size([1])

    @property
    def label_shape(self) -> torch.Size:
        return torch.Size([1])

    @classmethod
    def accepts_seed(cls) -> bool:
        return False

    def create_subset(self, indices: range) -> "_SmokeDataset":
        return _SmokeDataset([self.values[index] for index in indices])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.values[idx]
        return torch.tensor([value]), torch.tensor([value])

    def __len__(self) -> int:
        return len(self.values)


def test_eval_settings_keep_defaults_without_smoke_test() -> None:
    settings = eval_settings(
        smoke_test=False,
        default_n_samples=100,
        default_batch_size=10,
    )

    assert settings == (100, 10)


def test_eval_settings_reduce_work_for_smoke_test() -> None:
    settings = eval_settings(
        smoke_test=True,
        default_n_samples=100,
        default_batch_size=10,
    )

    assert settings == (4, 2)


def test_eval_settings_use_two_batches_for_smoke_test() -> None:
    only_n_samples, batch_size = eval_settings(
        smoke_test=True,
        default_n_samples=100,
        default_batch_size=8,
    )

    assert only_n_samples / batch_size == 2


def test_training_batch_size_keeps_default_without_smoke_test() -> None:
    batch_size = training_batch_size(
        smoke_test=False,
        default_batch_size=128,
    )

    assert batch_size == 128


def test_training_batch_size_reduces_work_for_smoke_test() -> None:
    batch_size = training_batch_size(
        smoke_test=True,
        default_batch_size=128,
    )

    assert batch_size == 2


def test_training_subset_keeps_all_rows_without_smoke_test() -> None:
    X_train = torch.arange(60).reshape(20, 3)
    y_train = torch.arange(20)

    X_subset, y_subset = training_subset(
        X_train,
        y_train,
        smoke_test=False,
    )

    assert torch.equal(X_subset, X_train)
    assert torch.equal(y_subset, y_train)


def test_training_subset_reduces_rows_for_smoke_test() -> None:
    X_train = torch.arange(60).reshape(20, 3)
    y_train = torch.arange(20)

    X_subset, y_subset = training_subset(
        X_train,
        y_train,
        smoke_test=True,
    )

    assert torch.equal(X_subset, X_train[:10])
    assert torch.equal(y_subset, y_train[:10])


def test_dataset_subset_reduces_to_two_batches_for_smoke_test() -> None:
    dataset = _SmokeDataset(list(range(20)))

    subset = dataset_subset(dataset, smoke_test=True)

    assert len(subset) == 4
    assert subset.values == [0, 1, 2, 3]


def test_dataset_subset_keeps_dataset_without_smoke_test() -> None:
    dataset = _SmokeDataset(list(range(20)))

    subset = dataset_subset(dataset, smoke_test=False)

    assert subset is dataset


def _supervised_pretraining_config(
    *, smoke_test: bool
) -> SupervisedPretrainingConfig:
    return SupervisedPretrainingConfig(
        train_dataset_bundle_path="train.bundle",
        val_dataset_bundle_path="val.bundle",
        classifier_bundle_path="classifier.bundle",
        save_path="model.bundle",
        initializer=InitializerConfig(class_name="ColdInitializer", kwargs={}),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        device="cpu",
        seed=0,
        smoke_test=smoke_test,
        supervised_learning=SupervisedLearningConfig(
            batch_size=128,
            max_epochs=100,
            checkpoint_earliest_batch=0,
            early_stopping_min_batches=0,
            early_stopping_patience=50,
            early_stopping_min_delta=1e-3,
            val_check_interval=10,
            limit_train_batches=None,
            limit_val_batches=None,
        ),
    )


def test_limit_supervised_learning_keeps_config_without_smoke_test() -> None:
    cfg = _supervised_pretraining_config(smoke_test=False)

    assert limit_supervised_learning(cfg) == cfg


def test_limit_supervised_learning_trains_one_short_epoch_for_smoke_test() -> (
    None
):
    cfg = _supervised_pretraining_config(smoke_test=True)

    limited = limit_supervised_learning(cfg).supervised_learning

    assert (
        limited.max_epochs,
        limited.limit_train_batches,
        limited.limit_val_batches,
    ) == (1, 2, 2)
    assert limited.batch_size == cfg.supervised_learning.batch_size
    assert cfg.supervised_learning.max_epochs == 100


@dataclass
class _StopCriteria:
    nepochs: int
    patience: int


@dataclass(frozen=True, kw_only=True)
class _MethodPretrainConfig(PretrainingContract):
    batch_size: int
    selector: _StopCriteria
    classifier: _StopCriteria


def _method_config(*, smoke_test: bool) -> _MethodPretrainConfig:
    return _MethodPretrainConfig(
        train_dataset_bundle_path="train.bundle",
        val_dataset_bundle_path="val.bundle",
        classifier_bundle_path="classifier.bundle",
        save_path="model.bundle",
        initializer=InitializerConfig(class_name="ColdInitializer", kwargs={}),
        unmasker=UnmaskerConfig(class_name="DirectUnmasker", kwargs={}),
        dataset_key="cube",
        device="cpu",
        seed=0,
        smoke_test=smoke_test,
        batch_size=128,
        selector=_StopCriteria(nepochs=100, patience=10),
        classifier=_StopCriteria(nepochs=100, patience=10),
    )


def test_limit_replaces_the_given_fields_for_smoke_test() -> None:
    cfg = _method_config(smoke_test=True)

    limited = limit(cfg, batch_size=2)

    assert limited.batch_size == 2
    assert {**vars(limited), "batch_size": 128} == vars(cfg)
    assert cfg.batch_size == 128


def test_limit_keeps_config_without_smoke_test() -> None:
    cfg = _method_config(smoke_test=False)

    assert limit(cfg, batch_size=2) is cfg


def test_limit_replaces_only_the_named_fields_of_a_nested_config() -> None:
    cfg = _method_config(smoke_test=True)

    limited = limit(cfg, selector={"nepochs": 1})

    assert limited.selector == _StopCriteria(nepochs=1, patience=10)
    assert limited.classifier == _StopCriteria(nepochs=100, patience=10)
    assert cfg.selector == _StopCriteria(nepochs=100, patience=10)
