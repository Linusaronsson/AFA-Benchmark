import pytest
import torch

from afabench.missing_values.config import TrainingMissingnessConfig
from afabench.missing_values.masking import FittedMissingnessMechanism


@pytest.fixture
def features() -> torch.Tensor:
    return torch.randn(8000, 12, generator=torch.Generator().manual_seed(7))


@pytest.mark.parametrize(
    "mechanism",
    ["mcar", "mar", "mnar_logistic", "mnar_self"],
)
def test_mechanism_is_reproducible_and_never_empties_rows(
    features: torch.Tensor,
    mechanism: str,
) -> None:
    cfg = TrainingMissingnessConfig(mechanism=mechanism, p=0.5)
    fitted = FittedMissingnessMechanism.fit(features, cfg, seed=11)

    first = fitted.sample_observed_mask(features, seed=12)
    second = fitted.sample_observed_mask(features, seed=12)

    assert torch.equal(first, second)
    assert first.any(dim=1).all()


@pytest.mark.parametrize(
    "mechanism",
    ["mcar", "mnar_logistic", "mnar_self"],
)
def test_global_missing_rate_matches_requested_probability(
    features: torch.Tensor,
    mechanism: str,
) -> None:
    cfg = TrainingMissingnessConfig(mechanism=mechanism, p=0.3)
    fitted = FittedMissingnessMechanism.fit(features, cfg, seed=3)
    observed = fitted.sample_observed_mask(features, seed=4)

    assert float((~observed).float().mean()) == pytest.approx(0.3, abs=0.02)


def test_mar_keeps_predictor_features_observed(
    features: torch.Tensor,
) -> None:
    cfg = TrainingMissingnessConfig(mechanism="mar", p=0.5, p_obs=0.3)
    fitted = FittedMissingnessMechanism.fit(features, cfg, seed=5)
    observed = fitted.sample_observed_mask(features, seed=6)

    assert observed[:, fitted.input_indices].all()
    target_missing_rate = (~observed[:, fitted.target_indices]).float().mean()
    assert float(target_missing_rate) == pytest.approx(0.5, abs=0.02)


def test_none_returns_complete_support(features: torch.Tensor) -> None:
    fitted = FittedMissingnessMechanism.fit(
        features,
        TrainingMissingnessConfig(),
        seed=0,
    )

    assert fitted.sample_observed_mask(features, seed=1).all()


@pytest.mark.parametrize(
    "mechanism",
    ["mcar", "mar", "mnar_logistic", "mnar_self"],
)
def test_grouped_features_share_one_missingness_indicator(
    features: torch.Tensor,
    mechanism: str,
) -> None:
    group_ids = torch.tensor([0, 0, 0, 1, 1, 2, 3, 4, 5, 6, 7, 8])
    fitted = FittedMissingnessMechanism.fit(
        features,
        TrainingMissingnessConfig(mechanism=mechanism, p=0.5),
        seed=21,
        group_ids=group_ids,
    )

    observed = fitted.sample_observed_mask(features, seed=22)

    assert torch.equal(observed[:, 0], observed[:, 1])
    assert torch.equal(observed[:, 0], observed[:, 2])
    assert torch.equal(observed[:, 3], observed[:, 4])


def test_grouped_mcar_probability_is_per_acquisition_group(
    features: torch.Tensor,
) -> None:
    group_ids = torch.tensor([0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7])
    fitted = FittedMissingnessMechanism.fit(
        features,
        TrainingMissingnessConfig(mechanism="mcar", p=0.7),
        seed=31,
        group_ids=group_ids,
    )

    observed = fitted.sample_observed_mask(features, seed=32)

    context_available = observed[:, :5].all(dim=1).float().mean()
    assert float(context_available) == pytest.approx(0.3, abs=0.03)


def test_noncontiguous_group_labels_are_canonicalized(
    features: torch.Tensor,
) -> None:
    group_ids = torch.tensor([4, 4, 9, 9, 9, 20, 21, 22, 23, 24, 25, 26])
    fitted = FittedMissingnessMechanism.fit(
        features,
        TrainingMissingnessConfig(mechanism="mcar", p=0.5),
        seed=41,
        group_ids=group_ids,
    )

    assert torch.equal(
        fitted.group_ids,
        torch.tensor([0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8]),
    )
