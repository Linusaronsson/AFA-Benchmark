from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification

from afabench.common.bundle import load_bundle, save_bundle
from afabench.common.classifiers import WrappedMALearnClassifier
from afabench.missing_values.malearn import (
    MADTClassifier,
    MAGBTClassifier,
    MALassoClassifier,
    MARFClassifier,
)


@pytest.fixture
def multiclass_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=160,
        n_features=12,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=9,
    )
    rng = np.random.default_rng(17)
    M = (rng.uniform(size=X.shape) < 0.25).astype(np.int8)
    return X, y, M


@pytest.mark.parametrize(
    ("model", "model_name"),
    [
        (MALassoClassifier(alpha=1.0, beta=1.0, random_state=3), "malasso"),
        (MADTClassifier(max_depth=4, alpha=1.0, random_state=3), "madt"),
        (
            MARFClassifier(
                n_estimators=8,
                max_depth=4,
                alpha=1.0,
                random_state=3,
                n_jobs=1,
            ),
            "marf",
        ),
        (
            MAGBTClassifier(
                n_estimators=10,
                max_depth=3,
                alpha=1.0,
                random_state=3,
            ),
            "magbt",
        ),
    ],
)
def test_wrapped_ma_classifier_predicts_probabilities(
    model: object,
    model_name: str,
    multiclass_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, M = multiclass_data
    model.fit(X, y, M=M)

    wrapped = WrappedMALearnClassifier(
        model=model,
        model_name=model_name,
        n_classes=3,
        device=torch.device("cpu"),
    )

    X_t = torch.tensor(X[:20], dtype=torch.float32)
    mask_t = torch.tensor(~M[:20].astype(bool), dtype=torch.bool)
    X_masked = X_t.clone()
    X_masked[~mask_t] = 0.0

    probs = wrapped(
        masked_features=X_masked,
        feature_mask=mask_t,
        feature_shape=torch.Size([X.shape[1]]),
    )

    assert probs.shape == (20, 3)
    assert torch.all(torch.isfinite(probs))
    row_sums = probs.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_wrapped_ma_classifier_bundle_roundtrip(
    tmp_path: Path,
    multiclass_data: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    X, y, M = multiclass_data
    model = MALassoClassifier(alpha=1.0, beta=1.0, random_state=5)
    model.fit(X, y, M=M)

    wrapped = WrappedMALearnClassifier(
        model=model,
        model_name="malasso",
        n_classes=3,
        device=torch.device("cpu"),
    )

    bundle_path = tmp_path / "ma_classifier.bundle"
    save_bundle(wrapped, bundle_path, metadata={"test": True})

    loaded, _manifest = load_bundle(bundle_path, device=torch.device("cpu"))
    loaded = cast("WrappedMALearnClassifier", loaded)

    X_t = torch.tensor(X[:10], dtype=torch.float32)
    mask_t = torch.tensor(~M[:10].astype(bool), dtype=torch.bool)
    X_masked = X_t.clone()
    X_masked[~mask_t] = 0.0

    out_original = wrapped(
        masked_features=X_masked,
        feature_mask=mask_t,
        feature_shape=torch.Size([X.shape[1]]),
    )
    out_loaded = loaded(
        masked_features=X_masked,
        feature_mask=mask_t,
        feature_shape=torch.Size([X.shape[1]]),
    )

    assert torch.allclose(out_original, out_loaded, atol=1e-7)
