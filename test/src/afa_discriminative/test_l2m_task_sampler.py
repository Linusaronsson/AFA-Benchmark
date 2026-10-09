"""L2M's task prior sampler (CONTEXT.md: task prior, context set)."""

import pytest
import torch

from afabench.components.methods.discriminative.l2m.task_sampler import (
    sample_task,
)


def test_real_feature_source_draws_instances_of_the_pool() -> None:
    pool = torch.arange(5 * 3, dtype=torch.float32).reshape(5, 3)

    task = sample_task(
        "real",
        n_features=3,
        sequence_length=4,
        label_shape=torch.Size([2]),
        missingness_cap=0.0,
        feature_pool=pool,
        seed=0,
    )

    assert task.features.shape == (4, 3)
    pool_instances = {tuple(instance.tolist()) for instance in pool}
    instances = [tuple(instance.tolist()) for instance in task.features]
    assert set(instances) <= pool_instances
    # Without replacement within a task.
    assert len(set(instances)) == len(instances)


def test_synthetic_feature_source_stays_within_the_uniform_box() -> None:
    task = sample_task(
        "synthetic",
        n_features=6,
        sequence_length=50,
        label_shape=torch.Size([2]),
        missingness_cap=0.0,
        seed=0,
    )

    assert task.features.shape == (50, 6)
    assert torch.all(task.features >= -2.0)
    assert torch.all(task.features <= 2.0)


def test_labels_have_the_configured_label_shape() -> None:
    task = sample_task(
        "synthetic",
        n_features=4,
        sequence_length=10,
        label_shape=torch.Size([5]),
        seed=0,
    )

    assert task.labels.shape == (10, 5)
    assert torch.equal(task.labels.sum(dim=-1), torch.ones(10))
    assert torch.all((task.labels == 0.0) | (task.labels == 1.0))


def test_missingness_cap_zero_leaves_every_feature_observed() -> None:
    for seed in range(5):
        task = sample_task(
            "synthetic",
            n_features=8,
            sequence_length=20,
            label_shape=torch.Size([2]),
            missingness_cap=0.0,
            seed=seed,
        )
        assert bool(task.feature_mask.all())


def test_missing_frequency_per_feature_never_exceeds_the_cap() -> None:
    cap = 0.3
    n_features = 5
    observed_counts = torch.zeros(n_features)
    total = 0
    for seed in range(200):
        task = sample_task(
            "synthetic",
            n_features=n_features,
            sequence_length=5,
            label_shape=torch.Size([2]),
            missingness_cap=cap,
            seed=seed,
        )
        observed_counts += task.feature_mask.sum(dim=0).float()
        total += task.feature_mask.shape[0]

    missing_frequency = 1.0 - observed_counts / total
    assert torch.all(missing_frequency <= cap + 0.02)


def test_same_seed_reproduces_task_different_seed_differs() -> None:
    kwargs = {
        "feature_source": "synthetic",
        "n_features": 4,
        "sequence_length": 10,
        "label_shape": torch.Size([2]),
        "missingness_cap": 0.3,
    }

    first = sample_task(**kwargs, seed=42)
    repeat = sample_task(**kwargs, seed=42)
    other = sample_task(**kwargs, seed=43)

    assert torch.equal(first.features, repeat.features)
    assert torch.equal(first.labels, repeat.labels)
    assert torch.equal(first.feature_mask, repeat.feature_mask)
    assert not torch.equal(first.features, other.features)


def test_real_feature_source_requires_a_feature_pool() -> None:
    with pytest.raises(ValueError, match="feature_pool"):
        sample_task(
            "real",
            n_features=3,
            sequence_length=4,
            label_shape=torch.Size([2]),
            seed=0,
        )


def test_synthetic_feature_source_rejects_a_feature_pool() -> None:
    with pytest.raises(ValueError, match="feature_pool"):
        sample_task(
            "synthetic",
            n_features=3,
            sequence_length=4,
            label_shape=torch.Size([2]),
            feature_pool=torch.zeros(5, 3),
            seed=0,
        )


def test_real_feature_source_rejects_a_pool_smaller_than_the_sequence() -> (
    None
):
    with pytest.raises(ValueError, match="sequence_length"):
        sample_task(
            "real",
            n_features=3,
            sequence_length=4,
            label_shape=torch.Size([2]),
            feature_pool=torch.zeros(3, 3),
            seed=0,
        )


def test_rejects_missingness_cap_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="missingness_cap"):
        sample_task(
            "synthetic",
            n_features=3,
            sequence_length=4,
            label_shape=torch.Size([2]),
            missingness_cap=1.5,
            seed=0,
        )


def test_rejects_a_label_shape_with_fewer_than_two_classes() -> None:
    with pytest.raises(ValueError, match="label_shape"):
        sample_task(
            "synthetic",
            n_features=3,
            sequence_length=4,
            label_shape=torch.Size([1]),
            seed=0,
        )
