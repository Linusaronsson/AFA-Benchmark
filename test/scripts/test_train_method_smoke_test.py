import torch

from afabench.training.smoke_test import eval_settings, training_subset


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

    assert settings == (10, 2)


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
