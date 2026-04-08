import torch

from afabench.common.datasets.datasets import XORNoisyShortcutDataset
from afabench.common.initializers.xor_noisy_shortcut_initializer import (
    XORNoisyShortcutInitializer,
)


def _make_features() -> tuple[torch.Tensor, torch.Size]:
    dataset = XORNoisyShortcutDataset(
        n_samples=4096,
        seed=0,
        shortcut_accuracy=0.51,
    )
    features, _ = dataset.get_all_data()
    return features, dataset.feature_shape


def test_initializer_starts_cold_and_only_blocks_feature_two() -> None:
    features, feature_shape = _make_features()
    initializer = XORNoisyShortcutInitializer(mechanism="mcar", p=0.5)
    initializer.set_seed(7)

    observed = initializer.initialize(
        features=features,
        feature_shape=feature_shape,
    )
    forbidden = initializer.get_training_forbidden_mask(observed)

    assert not observed.any()
    assert torch.equal(forbidden[:, 0], torch.zeros_like(forbidden[:, 0]))
    assert torch.equal(forbidden[:, 2], torch.zeros_like(forbidden[:, 2]))

    forbidden_selection = initializer.get_forbidden_selection_mask(
        observed,
        feature_shape,
    )
    assert torch.equal(forbidden_selection[:, 1], forbidden[:, 1])
    assert torch.equal(
        forbidden_selection[:, 0],
        torch.zeros_like(forbidden_selection[:, 0]),
    )
    assert torch.equal(
        forbidden_selection[:, 2],
        torch.zeros_like(forbidden_selection[:, 2]),
    )


def test_mcar_blocks_feature_two_at_target_rate() -> None:
    features, feature_shape = _make_features()
    initializer = XORNoisyShortcutInitializer(mechanism="mcar", p=0.5)
    initializer.set_seed(1)

    observed = initializer.initialize(
        features=features,
        feature_shape=feature_shape,
    )
    forbidden = initializer.get_training_forbidden_mask(observed)
    frac_blocked = forbidden[:, 1].float().mean().item()

    assert abs(frac_blocked - 0.5) < 0.05


def test_mar_blocking_depends_on_x1() -> None:
    features, feature_shape = _make_features()
    initializer = XORNoisyShortcutInitializer(
        mechanism="mar",
        p=0.5,
        structured_shift=0.25,
    )
    initializer.set_seed(2)

    observed = initializer.initialize(
        features=features,
        feature_shape=feature_shape,
    )
    forbidden = initializer.get_training_forbidden_mask(observed)

    x1 = features[:, 0] < 0.5
    high = forbidden[x1, 1].float().mean().item()
    low = forbidden[~x1, 1].float().mean().item()

    assert high > low + 0.25


def test_mnar_blocking_depends_on_x2() -> None:
    features, feature_shape = _make_features()
    initializer = XORNoisyShortcutInitializer(
        mechanism="mnar",
        p=0.5,
        structured_shift=0.25,
    )
    initializer.set_seed(3)

    observed = initializer.initialize(
        features=features,
        feature_shape=feature_shape,
    )
    forbidden = initializer.get_training_forbidden_mask(observed)

    x2 = features[:, 1] < 0.5
    high = forbidden[x2, 1].float().mean().item()
    low = forbidden[~x2, 1].float().mean().item()

    assert high > low + 0.25
