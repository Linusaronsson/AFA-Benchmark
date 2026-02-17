import numpy as np
import pytest
from sklearn.datasets import make_classification

from afabench.missing_values.malearn import (
    MADTClassifier,
    MAGBTClassifier,
    MALassoClassifier,
    MARFClassifier,
)


@pytest.fixture
def binary_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=180,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=7,
    )
    rng = np.random.default_rng(13)
    M = (rng.uniform(size=X.shape) < 0.25).astype(int)
    return X, y, M


@pytest.mark.parametrize(
    "model",
    [
        MALassoClassifier(alpha=1.0, beta=1.0, random_state=1),
        MADTClassifier(max_depth=4, alpha=1.0, random_state=1),
        MARFClassifier(
            n_estimators=6,
            max_depth=4,
            alpha=1.0,
            random_state=1,
            n_jobs=1,
        ),
        MAGBTClassifier(
            n_estimators=8,
            max_depth=2,
            alpha=1.0,
            random_state=1,
        ),
    ],
)
def test_fit_predict_and_missingness_reliance(
    model: MALassoClassifier
    | MADTClassifier
    | MARFClassifier
    | MAGBTClassifier,
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, M = binary_data

    fitted = model.fit(X, y, M=M)
    assert fitted is model

    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)

    proba = model.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    assert proba.shape[1] == len(np.unique(y))

    rho = model.compute_missingness_reliance(X, M)
    assert 0.0 <= rho <= 1.0


def test_invalid_missingness_mask_shape_raises(
    binary_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, M = binary_data
    bad_M = M[:-1]
    model = MALassoClassifier(alpha=1.0, random_state=2)
    with pytest.raises(ValueError, match="samples in X and M"):
        model.fit(X, y, M=bad_M)
