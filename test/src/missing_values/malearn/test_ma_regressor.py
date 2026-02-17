import numpy as np
from sklearn.datasets import make_regression

from afabench.missing_values.malearn import MADTRegressor


def test_madt_regressor_smoke() -> None:
    X, y = make_regression(
        n_samples=120,
        n_features=10,
        n_informative=8,
        noise=0.1,
        random_state=11,
    )
    rng = np.random.default_rng(5)
    M = (rng.uniform(size=X.shape) < 0.2).astype(int)

    model = MADTRegressor(max_depth=4, alpha=1.0, random_state=3)
    model.fit(X, y, M=M)

    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)

    rho = model.compute_missingness_reliance(X, M)
    assert 0.0 <= rho <= 1.0
