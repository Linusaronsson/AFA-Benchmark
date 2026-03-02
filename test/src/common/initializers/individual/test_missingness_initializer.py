import pytest
import torch

from afabench.common.custom_types import Features
from afabench.common.initializers.missingness_initializer import (
    MissingnessInitializer,
)


@pytest.fixture
def tabular_features() -> tuple[Features, torch.Size]:
    """Tabular features: 200 samples, 10 features."""
    batch_size = 200
    feature_shape = torch.Size([10])
    features = torch.randn(batch_size, *feature_shape)
    return features, feature_shape


@pytest.mark.parametrize(
    "mechanism",
    ["mcar", "mar", "mnar_logistic", "mnar_self", "mnar_quantiles"],
)
def test_shape_and_dtype(
    mechanism: str,
    tabular_features: tuple[Features, torch.Size],
) -> None:
    """Each mechanism produces correct shape and bool dtype."""
    features, feature_shape = tabular_features
    init = MissingnessInitializer(mechanism=mechanism, p=0.3)
    init.set_seed(42)
    mask = init.initialize(features=features, feature_shape=feature_shape)
    assert mask.shape == features.shape
    assert mask.dtype == torch.bool


def test_invalid_mechanism_raises() -> None:
    """Unknown mechanism raises ValueError."""
    with pytest.raises(ValueError, match="Unknown mechanism"):
        MissingnessInitializer(mechanism="invalid", p=0.3)


def test_seed_determinism(
    tabular_features: tuple[Features, torch.Size],
) -> None:
    """Same seed produces identical masks."""
    features, feature_shape = tabular_features

    init1 = MissingnessInitializer(mechanism="mar", p=0.3)
    init1.set_seed(123)
    mask1 = init1.initialize(features=features, feature_shape=feature_shape)

    init2 = MissingnessInitializer(mechanism="mar", p=0.3)
    init2.set_seed(123)
    mask2 = init2.initialize(features=features, feature_shape=feature_shape)

    assert torch.equal(mask1, mask2)


def test_different_seeds_differ(
    tabular_features: tuple[Features, torch.Size],
) -> None:
    """Different seeds produce different masks."""
    features, feature_shape = tabular_features

    init1 = MissingnessInitializer(mechanism="mar", p=0.3)
    init1.set_seed(111)
    mask1 = init1.initialize(features=features, feature_shape=feature_shape)

    init2 = MissingnessInitializer(mechanism="mar", p=0.3)
    init2.set_seed(222)
    mask2 = init2.initialize(features=features, feature_shape=feature_shape)

    assert not torch.equal(mask1, mask2)


def test_mask_convention(
    tabular_features: tuple[Features, torch.Size],
) -> None:
    """True=observed (AFA convention), not True=missing."""
    features, feature_shape = tabular_features
    init = MissingnessInitializer(mechanism="mar", p=0.5)
    init.set_seed(42)
    mask = init.initialize(features=features, feature_shape=feature_shape)
    # With p=0.5, roughly half should be observed (True)
    frac_observed = mask.float().mean().item()
    assert 0.3 < frac_observed < 0.7, (
        f"Expected ~50% observed, got {frac_observed:.2%}"
    )


@pytest.mark.parametrize("p", [0.1, 0.3, 0.5])
def test_approximate_missingness_proportion(
    p: float,
    tabular_features: tuple[Features, torch.Size],
) -> None:
    """Overall missingness should match MAR mechanism expectations."""
    features, feature_shape = tabular_features
    p_obs = 0.3
    init = MissingnessInitializer(mechanism="mar", p=p, p_obs=p_obs)
    init.set_seed(42)
    mask = init.initialize(features=features, feature_shape=feature_shape)
    frac_missing = 1.0 - mask.float().mean().item()
    # For MAR, p is applied only to the non-always-observed variables.
    n_features = feature_shape.numel()
    n_obs = max(int(p_obs * n_features), 1)
    expected_missing = p * (n_features - n_obs) / n_features
    assert abs(frac_missing - expected_missing) < 0.1, (
        f"Expected ~{expected_missing:.2%} missing under MAR, "
        f"got {frac_missing:.2%}"
    )


def test_feature_shape_required() -> None:
    """feature_shape=None raises AssertionError."""
    features = torch.randn(10, 5)
    init = MissingnessInitializer(mechanism="mar", p=0.3)
    with pytest.raises(AssertionError, match="feature_shape"):
        init.initialize(features=features, feature_shape=None)


def test_forbidden_selection_mask_fraction_half(
    tabular_features: tuple[Features, torch.Size],
) -> None:
    """With acquirable_fraction=0.5, roughly half of missing features are forbidden."""
    features, feature_shape = tabular_features
    init = MissingnessInitializer(
        mechanism="mar", p=0.3, acquirable_fraction=0.5
    )
    init.set_seed(42)
    observed_mask = init.initialize(
        features=features, feature_shape=feature_shape
    )
    forbidden = init.get_forbidden_selection_mask(observed_mask, feature_shape)

    assert forbidden.dtype == torch.bool
    assert forbidden.shape == observed_mask.shape  # both flat in selection dim

    # Forbidden features must be a subset of missing features
    flat_observed = observed_mask.reshape(features.shape[0], -1)
    assert (forbidden & flat_observed).sum() == 0, (
        "Observed features should never be forbidden"
    )

    # Check approximate fraction
    flat_missing = ~flat_observed
    n_missing = flat_missing.sum().item()
    n_forbidden = forbidden.sum().item()
    if n_missing > 0:
        frac_forbidden = n_forbidden / n_missing
        assert 0.3 < frac_forbidden < 0.7, (
            f"Expected ~50% of missing features forbidden, got {frac_forbidden:.2%}"
        )


def test_forbidden_selection_mask_fraction_one(
    tabular_features: tuple[Features, torch.Size],
) -> None:
    """With acquirable_fraction=1.0, nothing is forbidden."""
    features, feature_shape = tabular_features
    init = MissingnessInitializer(
        mechanism="mar", p=0.3, acquirable_fraction=1.0
    )
    init.set_seed(42)
    observed_mask = init.initialize(
        features=features, feature_shape=feature_shape
    )
    forbidden = init.get_forbidden_selection_mask(observed_mask, feature_shape)
    assert forbidden.sum() == 0


def test_forbidden_selection_mask_fraction_zero(
    tabular_features: tuple[Features, torch.Size],
) -> None:
    """With acquirable_fraction=0.0, all missing features are forbidden."""
    features, feature_shape = tabular_features
    init = MissingnessInitializer(
        mechanism="mar", p=0.3, acquirable_fraction=0.0
    )
    init.set_seed(42)
    observed_mask = init.initialize(
        features=features, feature_shape=feature_shape
    )
    forbidden = init.get_forbidden_selection_mask(observed_mask, feature_shape)

    flat_missing = ~observed_mask.reshape(features.shape[0], -1)
    assert torch.equal(forbidden, flat_missing), (
        "All missing features should be forbidden when acquirable_fraction=0"
    )


def test_forbidden_mask_seed_determinism(
    tabular_features: tuple[Features, torch.Size],
) -> None:
    """Same seed produces identical forbidden masks."""
    features, feature_shape = tabular_features

    init1 = MissingnessInitializer(
        mechanism="mar", p=0.3, acquirable_fraction=0.5
    )
    init1.set_seed(42)
    obs1 = init1.initialize(features=features, feature_shape=feature_shape)
    fb1 = init1.get_forbidden_selection_mask(obs1, feature_shape)

    init2 = MissingnessInitializer(
        mechanism="mar", p=0.3, acquirable_fraction=0.5
    )
    init2.set_seed(42)
    obs2 = init2.initialize(features=features, feature_shape=feature_shape)
    fb2 = init2.get_forbidden_selection_mask(obs2, feature_shape)

    assert torch.equal(fb1, fb2)
