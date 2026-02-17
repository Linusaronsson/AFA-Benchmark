# ruff: noqa
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.utils import check_array


@contextmanager
def patch_missingness_mask(estimator, M: np.ndarray | None):
    estimator._missingness_mask = M
    try:
        yield estimator
    finally:
        delattr(estimator, "_missingness_mask")


def check_missingness_mask(
    M: np.ndarray | None, X: np.ndarray
) -> np.ndarray | None:
    if M is None:
        return None
    M = check_array(M)
    if M.shape[0] != X.shape[0]:
        raise ValueError("The number of samples in X and M must be the same.")
    return M


def _tree_to_dataframe(tree, idx: int = 0) -> pd.DataFrame:
    n_nodes = tree.node_count
    tree_index = np.full(n_nodes, idx)
    node_ids = [f"{idx}-{i}" for i in range(n_nodes)]
    features = ["Leaf" if f == -2 else f"f{f}" for f in tree.feature]
    thresholds = tree.threshold
    children_left = [
        f"{idx}-{c}" if c > -1 else np.nan for c in tree.children_left
    ]
    children_right = [
        f"{idx}-{c}" if c > -1 else np.nan for c in tree.children_right
    ]

    return pd.DataFrame(
        {
            "Tree": tree_index,
            "ID": node_ids,
            "Feature": features,
            "Split": thresholds,
            "Yes": children_left,
            "No": children_right,
        }
    )


def _ensemble_missingness_reliance_from_df(
    df: pd.DataFrame,
    X: np.ndarray,
    M: np.ndarray,
    equality_left: bool = True,
) -> float:
    n_trees = int(df["Tree"].max()) + 1
    children = dict(zip(df["ID"], zip(df["Yes"], df["No"], strict=False)))
    features = dict(zip(df["ID"], df["Feature"], strict=False))
    thresholds = dict(zip(df["ID"], df["Split"], strict=False))

    n_samples, n_features = X.shape
    feat_id = {f"f{i}": i for i in range(n_features)}

    def check_missing(node: str, I: np.ndarray) -> list[int]:
        f = features[node]

        if f == "Leaf" or len(I) == 0:
            return []

        fid = feat_id[f]

        is_missing = M[I, fid].astype(bool)
        I_na = I[is_missing]
        I_o = I[~is_missing]

        left, right = children[node]
        if equality_left:
            I_l = I_o[X[I_o, fid] <= thresholds[node]]
            I_r = I_o[X[I_o, fid] > thresholds[node]]
        else:
            I_l = I_o[X[I_o, fid] < thresholds[node]]
            I_r = I_o[X[I_o, fid] >= thresholds[node]]

        miss_l = check_missing(left, I_l)
        miss_r = check_missing(right, I_r)

        return list(I_na) + miss_l + miss_r

    miss_rows = []
    for tree in range(n_trees):
        root_node = f"{tree}-0"
        miss_feat = check_missing(root_node, np.arange(n_samples))
        miss_rows.extend(miss_feat)

    return len(np.unique(miss_rows)) / n_samples


def get_ensemble_missingness_reliance(
    ensemble,
    X: np.ndarray,
    M: np.ndarray,
) -> float:
    if hasattr(ensemble, "estimators_"):
        estimators = ensemble.estimators_
        if isinstance(estimators, np.ndarray):
            estimators = estimators.ravel()
        dfs = [
            _tree_to_dataframe(est.tree_, i)
            for i, est in enumerate(estimators)
        ]
        df = pd.concat(dfs, ignore_index=True)
        return _ensemble_missingness_reliance_from_df(df, X, M)

    msg = f"Unknown ensemble model: {ensemble.__class__.__name__}."
    raise ValueError(msg)


def get_dt_missingness_reliance(
    dt,
    X: np.ndarray,
    M: np.ndarray,
) -> float:
    df = _tree_to_dataframe(dt.tree_)
    return _ensemble_missingness_reliance_from_df(df, X, M)


def get_lm_missingness_reliance(
    lm,
    X: np.ndarray,
    M: np.ndarray,
    reduction: str = "max",
) -> float:
    del X
    if reduction not in {"mean", "max"}:
        raise ValueError("Invalid reduction method.")
    if not hasattr(lm, "coef_"):
        msg = (
            "The model must have coefficients to compute missingness reliance."
        )
        raise ValueError(msg)
    beta = lm.coef_.ravel()
    if reduction == "mean":
        return np.mean(M * (beta != 0).astype(int), axis=0).mean()
    return np.max(M * (beta != 0).astype(int), axis=1).mean()
