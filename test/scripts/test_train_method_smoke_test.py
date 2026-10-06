import torch

from afabench.training.config import SupervisedLearningConfig
from afabench.training.smoke_test import (
    dataset_subset,
    eval_settings,
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


def _supervised_learning_config() -> SupervisedLearningConfig:
    return SupervisedLearningConfig(
        batch_size=128,
        max_epochs=100,
        checkpoint_earliest_batch=0,
        early_stopping_min_batches=0,
        early_stopping_patience=50,
        early_stopping_min_delta=1e-3,
        val_check_interval=10,
        limit_train_batches=None,
        limit_val_batches=None,
    )


def test_limit_supervised_learning_keeps_config_without_smoke_test() -> None:
    cfg = _supervised_learning_config()

    assert limit_supervised_learning(cfg, smoke_test=False) == cfg


def test_limit_supervised_learning_trains_one_short_epoch_for_smoke_test() -> (
    None
):
    cfg = _supervised_learning_config()

    limited = limit_supervised_learning(cfg, smoke_test=True)

    assert (
        limited.max_epochs,
        limited.limit_train_batches,
        limited.limit_val_batches,
    ) == (1, 2, 2)
    assert limited.batch_size == cfg.batch_size
    assert cfg.max_epochs == 100
