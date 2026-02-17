# ruff: noqa
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

import numpy as np
from sklearn.base import _fit_context
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble._forest import (
    _get_n_samples_bootstrap,
    _parallel_build_trees,
    ForestClassifier,
)
from sklearn.ensemble._gb import (
    _update_terminal_regions,
    BaseGradientBoosting,
    set_huber_delta,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree._tree import DTYPE
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import HasMethods, Interval, StrOptions
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn._loss.loss import AbsoluteError, HuberLoss, PinballLoss

from afabench.missing_values.malearn.base import (
    BaseMADT,
    ClassifierMixin,
    MALassoMixin,
)
from afabench.missing_values.malearn.criterion import (
    entropy,
    gini_impurity,
    gini_scorer,
    info_gain_scorer,
)
from afabench.missing_values.malearn.missingness_utils import (
    check_missingness_mask,
    get_ensemble_missingness_reliance,
    patch_missingness_mask,
)
from afabench.missing_values.malearn.regression import MADTRegressor

CRITERIA_CLF = {"info_gain": info_gain_scorer, "gini": gini_scorer}


class MALassoClassifier(MALassoMixin, ClassifierMixin, LogisticRegression):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        random_state: int | None = None,
        solver: str = "liblinear",
    ):
        C = 1 / (2 * alpha)
        super().__init__(
            penalty="l1", C=C, random_state=random_state, solver=solver
        )
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y, M=None):
        if M is not None:
            M = check_missingness_mask(M, X)
            X = self._transform_input(X, M)
        return super().fit(X, y)

    def set_params(self, **params):
        super().set_params(**params)
        self.C = 1 / (2 * self.alpha)


class MADTClassifier(ClassifierMixin, BaseMADT):
    """Missingness-avoiding decision tree classifier."""

    _parameter_constraints: dict = {
        **BaseMADT._parameter_constraints,
        "criterion": [StrOptions({"info_gain", "gini"})],
    }

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 3,
        random_state: int | None = None,
        alpha: float = 1.0,
        compute_rho_per_node: bool = False,
        ccp_alpha: float = 0.0,
    ):
        super().__init__(
            max_depth, random_state, alpha, compute_rho_per_node, ccp_alpha
        )
        self.criterion = criterion

    def _get_node_value(self, y, sample_weight):
        node_value = np.zeros(self.n_classes_, dtype=float)
        for class_label, weight in zip(y, sample_weight):
            node_value[class_label] += weight
        node_value /= np.sum(sample_weight)
        return node_value

    def _get_node_impurity(self, node_value, y, sample_weight):
        del y, sample_weight
        return (
            gini_impurity(node_value)
            if self.criterion == "gini"
            else entropy(node_value)
        )

    def _is_homogeneous(self, node_value, y, sample_weight):
        del y, sample_weight
        return max(node_value) == 1.0

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, M=None, sample_weight=None):
        return super()._fit(X, y, M, sample_weight=sample_weight)

    def predict_proba(self, X, check_input=True):
        return self.predict(X, check_input, return_proba=True)

    def _best_split(self, X, y, feature, sample_weight, M=None):
        sorted_indices = np.argsort(X[:, feature])
        X_sorted = X[sorted_indices, feature]
        y_sorted = y[sorted_indices]
        sample_weight_sorted = sample_weight[sorted_indices]

        low_distr = np.zeros(self.n_classes_, dtype=float)
        high_distr = np.zeros(self.n_classes_, dtype=float)

        for class_label, weight in zip(y_sorted, sample_weight_sorted):
            high_distr[class_label] += weight

        n_low = 0.0
        n_high = np.sum(sample_weight)

        max_score = -np.inf
        i_max_score = None

        n = len(y)
        for i in range(n - 1):
            yi = y_sorted[i]
            wi = sample_weight_sorted[i]

            low_distr[yi] += wi
            high_distr[yi] -= wi

            n_low += wi
            n_high -= wi

            if n_low == 0:
                continue

            if n_high == 0:
                break

            if np.isclose(X_sorted[i], X_sorted[i + 1]):
                continue

            criterion_function = CRITERIA_CLF[self.criterion]
            score = criterion_function(n_low, low_distr, n_high, high_distr)

            if score > max_score:
                max_score = score
                i_max_score = i

        if i_max_score is None:
            return -np.inf, None, None

        if M is not None:
            max_score -= self.alpha * np.mean(sample_weight * M[:, feature])

        split_threshold = 0.5 * (
            X_sorted[i_max_score] + X_sorted[i_max_score + 1]
        )
        return max_score, feature, split_threshold


class MARFClassifier(ClassifierMixin, ForestClassifier):
    """Missingness-avoiding random forest classifier."""

    _parameter_constraints: dict = {
        **ForestClassifier._parameter_constraints,
        **MADTClassifier._parameter_constraints,
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
            dict,
            list,
            None,
        ],
    }

    def __init__(
        self,
        n_estimators: int = 100,
        *,
        criterion: str = "gini",
        max_depth: int = 10,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: int = 0,
        class_weight=None,
        max_samples=None,
        alpha: float = 1.0,
        compute_rho_per_node: bool = False,
    ):
        super().__init__(
            estimator=MADTClassifier(),
            n_estimators=n_estimators,
            estimator_params=(
                "criterion",
                "max_depth",
                "alpha",
                "compute_rho_per_node",
            ),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=False,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        self.criterion = criterion
        self.max_depth = max_depth
        self.alpha = alpha
        self.compute_rho_per_node = compute_rho_per_node

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, M=None, sample_weight=None):
        X, y = self._validate_data(X, y, dtype=DTYPE)

        if M is not None:
            missing_values_in_feature_mask = check_missingness_mask(M, X)
        else:
            missing_values_in_feature_mask = None

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        y = np.atleast_1d(y)

        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        self._n_samples, self.n_outputs_ = y.shape

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        if not self.bootstrap and self.max_samples is not None:
            msg = (
                "`max_samples` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set `max_samples=None`."
            )
            raise ValueError(msg)
        if self.bootstrap:
            n_samples_bootstrap = _get_n_samples_bootstrap(
                n_samples=X.shape[0], max_samples=self.max_samples
            )
        else:
            n_samples_bootstrap = None

        self._n_samples_bootstrap = n_samples_bootstrap
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            msg = (
                "Out-of-bag estimation is only available if `bootstrap=True`."
            )
            raise ValueError(msg)

        random_state = check_random_state(self.random_state)

        trees = [
            self._make_estimator(append=False, random_state=random_state)
            for _ in range(self.n_estimators)
        ]

        self.estimators_ = Parallel(
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            prefer="threads",
        )(
            delayed(_parallel_build_trees)(
                t,
                self.bootstrap,
                X,
                y,
                sample_weight,
                i,
                len(trees),
                verbose=self.verbose,
                class_weight=self.class_weight,
                n_samples_bootstrap=n_samples_bootstrap,
                missing_values_in_feature_mask=missing_values_in_feature_mask,
            )
            for i, t in enumerate(trees)
        )

        if self.oob_score and not hasattr(self, "oob_score_"):
            if callable(self.oob_score):
                self._set_oob_score_and_attributes(
                    X, y, scoring_function=self.oob_score
                )
            else:
                self._set_oob_score_and_attributes(X, y)

        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_ensemble_missingness_reliance(self, X, M)


def predict_stages(estimators, X, scale, out):
    n_estimators = len(estimators)
    K = len(estimators[0])
    for i in range(n_estimators):
        for k in range(K):
            tree = estimators[i][k].tree_
            leaf_indices = tree.apply(X)
            out[:, k] += scale * tree.value[leaf_indices].ravel()


def predict_stage(estimators, stage, X, scale, out):
    return predict_stages(
        estimators=estimators[stage : stage + 1], X=X, scale=scale, out=out
    )


class BaseMAGBT(BaseGradientBoosting, metaclass=ABCMeta):
    _parameter_constraints: dict = {
        **MADTRegressor._parameter_constraints,
        "learning_rate": [Interval(Real, 0.0, None, closed="left")],
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "subsample": [Interval(Real, 0.0, 1.0, closed="right")],
        "verbose": ["verbose"],
        "warm_start": ["boolean"],
        "validation_fraction": [Interval(Real, 0.0, 1.0, closed="neither")],
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0.0, None, closed="left")],
        "alpha": [Interval(Real, 0.0, None, closed="neither")],
    }

    @abstractmethod
    def __init__(
        self,
        *,
        loss,
        learning_rate,
        n_estimators,
        subsample,
        max_depth,
        max_features,
        init,
        random_state,
        quantile=0.9,
        verbose=0,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        alpha=1.0,
        compute_rho_per_node=False,
    ):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth
        self.max_features = max_features
        self.init = init
        self.random_state = random_state
        self.quantile = quantile
        self.verbose = verbose
        self.warm_start = warm_start
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.alpha = alpha
        self.compute_rho_per_node = compute_rho_per_node

    def _init_state(self):
        self.init_ = self.init
        if self.init_ is None:
            if isinstance(self, ClassifierMixin):
                self.init_ = DummyClassifier(strategy="prior")
            elif isinstance(self._loss, (AbsoluteError, HuberLoss)):
                self.init_ = DummyRegressor(strategy="quantile", quantile=0.5)
            elif isinstance(self._loss, PinballLoss):
                self.init_ = DummyRegressor(
                    strategy="quantile", quantile=self.quantile
                )
            else:
                self.init_ = DummyRegressor(strategy="mean")

        self.estimators_ = np.empty(
            (self.n_estimators, self.n_trees_per_iteration_), dtype=object
        )
        self.train_score_ = np.zeros((self.n_estimators,), dtype=np.float64)
        if self.subsample < 1.0:
            self.oob_improvement_ = np.zeros(
                (self.n_estimators), dtype=np.float64
            )
            self.oob_scores_ = np.zeros((self.n_estimators), dtype=np.float64)
            self.oob_score_ = np.nan

    def _raw_predict(self, X):
        check_is_fitted(self)
        raw_predictions = self._raw_predict_init(X)
        predict_stages(
            self.estimators_, X, self.learning_rate, raw_predictions
        )
        return raw_predictions

    def _staged_raw_predict(self, X, check_input=True):
        if check_input:
            X = self._validate_data(X, dtype=DTYPE, order="C", reset=False)
        raw_predictions = self._raw_predict_init(X)
        for i in range(self.estimators_.shape[0]):
            predict_stage(
                self.estimators_, i, X, self.learning_rate, raw_predictions
            )
            yield raw_predictions.copy()

    def _fit_stage(
        self,
        i,
        X,
        y,
        raw_predictions,
        sample_weight,
        sample_mask,
        random_state,
        X_csc=None,
        X_csr=None,
    ):
        del X_csc, X_csr
        original_y = y

        if isinstance(self._loss, HuberLoss):
            set_huber_delta(
                loss=self._loss,
                y_true=y,
                raw_prediction=raw_predictions,
                sample_weight=sample_weight,
            )

        neg_gradient = -self._loss.gradient(
            y_true=y,
            raw_prediction=raw_predictions,
            sample_weight=None,
        )
        if neg_gradient.ndim == 1:
            neg_g_view = neg_gradient.reshape((-1, 1))
        else:
            neg_g_view = neg_gradient

        M = getattr(self, "_missingness_mask", None)

        for k in range(self.n_trees_per_iteration_):
            if self._loss.is_multiclass:
                y = np.array(original_y == k, dtype=np.float64)

            if M is not None:
                estimators = self.estimators_.ravel()
                last_estimator = next(
                    (x for x in np.flip(estimators) if x is not None), None
                )
                if last_estimator is not None:
                    miss_reliance = (
                        last_estimator.compute_missingness_reliance(
                            X, M, sample_mask=sample_mask, reduce=False
                        )
                    )
                    M = (M - miss_reliance).clip(min=0)

            tree = MADTRegressor(
                max_depth=self.max_depth,
                random_state=random_state,
                alpha=self.alpha,
                compute_rho_per_node=self.compute_rho_per_node,
            )

            if self.subsample < 1.0:
                sample_weight = sample_weight * sample_mask.astype(np.float64)

            tree._fit(
                X,
                neg_g_view[:, k],
                M=M,
                sample_weight=sample_weight,
                check_input=False,
            )

            _update_terminal_regions(
                self._loss,
                tree.tree_,
                X,
                y,
                neg_g_view[:, k],
                raw_predictions,
                sample_weight,
                sample_mask,
                learning_rate=self.learning_rate,
                k=k,
            )

            self.estimators_[i, k] = tree

        self._missingness_mask = M

        return raw_predictions


class MAGBTClassifier(ClassifierMixin, BaseMAGBT, GradientBoostingClassifier):
    """Missingness-avoiding gradient boosting classifier."""

    _parameter_constraints: dict = {
        **BaseGradientBoosting._parameter_constraints,
        "loss": [StrOptions({"log_loss", "exponential"})],
        "init": [
            StrOptions({"zero"}),
            None,
            HasMethods(["fit", "predict_proba"]),
        ],
    }

    def __init__(
        self,
        *,
        loss="log_loss",
        learning_rate=0.1,
        n_estimators=100,
        subsample=1.0,
        max_depth=3,
        init=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        alpha=1.0,
        compute_rho_per_node=False,
    ):
        super().__init__(
            loss=loss,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            init=init,
            subsample=subsample,
            max_features=None,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            tol=tol,
            alpha=alpha,
            compute_rho_per_node=compute_rho_per_node,
        )

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, M=None, sample_weight=None, monitor=None):
        if M is not None and self.n_iter_no_change is not None:
            M, _ = train_test_split(
                M,
                random_state=self.random_state,
                test_size=self.validation_fraction,
                stratify=y,
            )
        with patch_missingness_mask(self, M):
            return super().fit(
                X, y, sample_weight=sample_weight, monitor=monitor
            )

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_ensemble_missingness_reliance(self, X, M)
