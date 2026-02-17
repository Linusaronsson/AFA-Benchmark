# ruff: noqa
from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real

import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.tree._tree import Tree, _build_pruned_tree_ccp, ccp_pruning_path
from sklearn.utils import Bunch, check_random_state
from sklearn.utils._param_validation import Interval
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import (
    _check_sample_weight,
    check_X_y,
    check_array,
    check_is_fitted,
)

from afabench.missing_values.malearn.missingness_utils import (
    check_missingness_mask,
    get_lm_missingness_reliance,
)
from afabench.missing_values.malearn.tree import (
    ExtendedTree,
    TreeNode,
    convert_to_sklearn_tree,
    update_tree_structure,
)


class ClassifierMixin:
    _estimator_type = "classifier"

    accepted_metrics = {
        "accuracy": {
            "function": metrics.accuracy_score,
            "requires_proba": False,
            "kwargs": {},
            "lower_is_better": False,
        },
        "roc_auc_ovr": {
            "function": metrics.roc_auc_score,
            "requires_proba": True,
            "kwargs": {"multi_class": "ovr", "average": "macro"},
            "lower_is_better": False,
        },
        "roc_auc_ovr_weighted": {
            "function": metrics.roc_auc_score,
            "requires_proba": True,
            "kwargs": {"multi_class": "ovr", "average": "weighted"},
            "lower_is_better": False,
        },
        "log_loss": {
            "function": metrics.log_loss,
            "requires_proba": True,
            "kwargs": {"normalize": True},
            "lower_is_better": True,
        },
    }

    def score(
        self,
        X,
        y,
        M=None,
        sample_weight=None,
        metric="accuracy",
        **predict_params,
    ):
        is_accepted_metric = (
            metric in self.accepted_metrics
            or metric == "missingness_reliance_score"
        )
        if not is_accepted_metric:
            msg = (
                f"Got invalid metric {metric}. Valid metrics are "
                "'missingness_reliance_score' and "
                f"{self.accepted_metrics.keys()}."
            )
            raise ValueError(msg)

        if metric == "missingness_reliance_score":
            if M is None:
                msg = (
                    "The metric 'missingness_reliance' requires the "
                    "missingness mask `M`."
                )
                raise ValueError(msg)
            return (-1) * self.compute_missingness_reliance(X, M)

        if self.accepted_metrics[metric]["requires_proba"]:
            yp = self.predict_proba(X, **predict_params)
            if yp.shape[1] == 2:
                yp = yp[:, 1]
        else:
            yp = self.predict(X, **predict_params)

        score_params = self.accepted_metrics[metric]["kwargs"].copy()
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight

        try:
            return self.accepted_metrics[metric]["function"](
                y, yp, **score_params
            )
        except ValueError:
            return np.nan

    def _more_tags(self):
        return {"requires_y": True}


class RegressorMixin:
    _estimator_type = "regressor"

    accepted_metrics = {
        "r2": {
            "function": metrics.r2_score,
            "kwargs": {},
            "lower_is_better": False,
        },
        "mean_squared_error": {
            "function": metrics.mean_squared_error,
            "kwargs": {},
            "lower_is_better": True,
        },
        "mean_absolute_error": {
            "function": metrics.mean_absolute_error,
            "kwargs": {},
            "lower_is_better": True,
        },
    }

    def score(
        self,
        X,
        y,
        M=None,
        sample_weight=None,
        metric="r2",
        **predict_params,
    ):
        is_accepted_metric = (
            metric in self.accepted_metrics
            or metric == "missingness_reliance_score"
        )
        if not is_accepted_metric:
            msg = (
                f"Got invalid metric {metric}. Valid metrics are "
                "'missingness_reliance_score' and "
                f"{self.accepted_metrics.keys()}."
            )
            raise ValueError(msg)

        if metric == "missingness_reliance_score":
            if M is None:
                msg = (
                    "The metric 'missingness_reliance' requires the "
                    "missingness mask `M`."
                )
                raise ValueError(msg)
            return (-1) * self.compute_missingness_reliance(X, M)

        yp = self.predict(X, **predict_params)

        score_params = self.accepted_metrics[metric]["kwargs"].copy()
        if sample_weight is not None:
            score_params["sample_weight"] = sample_weight

        return self.accepted_metrics[metric]["function"](y, yp, **score_params)

    def _more_tags(self):
        return {"requires_y": True}


class BaseMADT(BaseEstimator, metaclass=ABCMeta):
    _parameter_constraints: dict = {
        "max_depth": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "alpha": [Interval(Real, 0.0, None, closed="left")],
        "compute_rho_per_node": [bool],
        "ccp_alpha": [Interval(Real, 0.0, None, closed="left")],
    }

    def __init__(
        self,
        max_depth,
        random_state,
        alpha,
        compute_rho_per_node,
        ccp_alpha,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.random_state = random_state
        self.alpha = alpha
        self.compute_rho_per_node = compute_rho_per_node
        self.ccp_alpha = ccp_alpha

    def _fit(
        self,
        X,
        y,
        M=None,
        *,
        clear_X_M=True,
        check_input=True,
        sample_weight=None,
        missing_values_in_feature_mask=None,
        keep_original_tree=False,
    ):
        if check_input:
            X, y = check_X_y(X, y)

        if y.ndim > 1:
            y = y.ravel()

        if is_classifier(self):
            labels = unique_labels(y)
            n_labels = len(labels)
            if not np.all(labels == np.arange(n_labels)):
                msg = (
                    "Labels are expected to be in the range 0 to n_classes-1."
                )
                raise ValueError(msg)

        if missing_values_in_feature_mask is not None:
            if M is not None:
                warnings.warn(
                    "Both `M` and `missing_values_in_feature_mask` are "
                    "provided. `M` will be used.",
                    stacklevel=2,
                )
            else:
                M = missing_values_in_feature_mask

        M = check_missingness_mask(M, X)
        sample_weight = _check_sample_weight(sample_weight, X)

        random_state = check_random_state(self.random_state)

        self.n_features_in_ = X.shape[1]
        if is_classifier(self):
            self.classes_ = unique_labels(y)
            self.n_classes_ = len(self.classes_)

        self._split_features = []
        self.root_ = self._make_tree(X, y, M, sample_weight, 0, random_state)

        if self.compute_rho_per_node and M is not None:
            self.root_.X, self.root_.M = X, M
            self._compute_missingness_reliance_for_all_nodes(
                self.root_, clear_X_M
            )

        n_classes = self.n_classes_ if is_classifier(self) else 1
        self.tree_ = convert_to_sklearn_tree(
            self.root_, self.n_features_in_, n_classes
        )

        self._prune_tree()

        nodes_after_pruning = self.tree_.__getstate__()["nodes"]
        update_tree_structure(
            self.root_, nodes_after_pruning[0], nodes_after_pruning
        )

        if not keep_original_tree:
            self.tree_ = ExtendedTree(self.tree_, self.root_)

        return self

    def _prune_tree(self) -> None:
        check_is_fitted(self)

        if self.ccp_alpha == 0.0:
            return

        if is_classifier(self):
            n_classes = np.atleast_1d(self.n_classes_)
            pruned_tree = Tree(self.n_features_in_, n_classes, 1)
        else:
            pruned_tree = Tree(
                self.n_features_in_, np.array([1], dtype=np.intp), 1
            )
        _build_pruned_tree_ccp(pruned_tree, self.tree_, self.ccp_alpha)

        self.tree_ = pruned_tree

    def cost_complexity_pruning_path(self, X, y, M=None, sample_weight=None):
        est = clone(self).set_params(ccp_alpha=0.0)
        est._fit(X, y, M, sample_weight=sample_weight, keep_original_tree=True)
        return Bunch(**ccp_pruning_path(est.tree_))

    def _validate_X_predict(self, X, check_input=True):
        if check_input:
            X = check_array(X)
        n_features = X.shape[1]
        if n_features != self.n_features_in_:
            msg = (
                f"X has {n_features} features, but {self.n_features_in_} "
                "features were expected."
            )
            raise ValueError(msg)
        return X

    def _support_missing_values(self, X):
        del X
        return False

    def _compute_missing_values_in_feature_mask(self, X, estimator_name=None):
        del X, estimator_name
        return None

    def predict(self, X, check_input=True, M=None, return_proba=False):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        M = check_missingness_mask(M, X)
        if M is not None:
            yp = np.array([self.root_.predict(x, m) for x, m in zip(X, M)])
        else:
            yp = np.array([self.root_.predict(x) for x in X])
        if is_classifier(self) and not return_proba:
            return np.argmax(yp, axis=1)
        return yp

    def get_prediction_depth(self, X, M=None):
        check_is_fitted(self)
        X = self._validate_X_predict(X)
        M = check_missingness_mask(M, X)
        if M is not None:
            return np.array(
                [
                    self.root_.predict(x, m, return_depth=True)[-1]
                    for x, m in zip(X, M)
                ]
            )
        return np.array(
            [self.root_.predict(x, return_depth=True)[-1] for x in X]
        )

    def _make_tree(self, X, y, M, sample_weight, depth, random_state):
        node_value = self._get_node_value(y, sample_weight)
        node_impurity = self._get_node_impurity(node_value, y, sample_weight)
        n_node_samples = len(X)
        weighted_n_node_samples = np.sum(sample_weight)

        if depth >= self.max_depth or self._is_homogeneous(
            node_value, y, sample_weight
        ):
            return TreeNode(
                node_value,
                node_impurity,
                n_node_samples=n_node_samples,
                weighted_n_node_samples=weighted_n_node_samples,
                depth=depth,
            )

        _split_score, split_feature, split_threshold = max(
            self._get_split_candidates(
                X, y, M, sample_weight, depth, random_state
            )
        )
        self._split_features.append(split_feature)

        if split_feature is None:
            return TreeNode(
                node_value,
                node_impurity,
                n_node_samples=n_node_samples,
                weighted_n_node_samples=weighted_n_node_samples,
                depth=depth,
            )

        data_left, data_right = self._split_by_feature(
            X, y, M, sample_weight, split_feature, split_threshold
        )

        left_subtree = self._make_tree(*data_left, depth + 1, random_state)
        right_subtree = self._make_tree(*data_right, depth + 1, random_state)

        if self.compute_rho_per_node and M is not None:
            left_subtree.X, left_subtree.M = data_left[0], data_left[2]
            right_subtree.X, right_subtree.M = data_right[0], data_right[2]

        if left_subtree == right_subtree:
            return left_subtree

        return TreeNode(
            node_value,
            node_impurity,
            n_node_samples=n_node_samples,
            weighted_n_node_samples=weighted_n_node_samples,
            feature=split_feature,
            threshold=split_threshold,
            left_subtree=left_subtree,
            right_subtree=right_subtree,
            depth=depth,
        )

    def _split_by_feature(self, X, y, M, sw, feature, threshold):
        left = X[:, feature] <= threshold
        right = ~left
        Xl, Xr = X[left], X[right]
        yl, yr = y[left], y[right]
        Ml, Mr = (M[left], M[right]) if M is not None else (None, None)
        swl, swr = sw[left], sw[right]
        return (Xl, yl, Ml, swl), (Xr, yr, Mr, swr)

    def _get_features_along_decision_path(self, x, max_depth=None):
        node = self.root_
        if max_depth is None:
            max_depth = self.max_depth
        while node.depth <= max_depth and node.left_subtree is not None:
            yield node.feature
            if x[node.feature] <= node.threshold:
                node = node.left_subtree
            else:
                node = node.right_subtree

    def _compute_missingness_reliance_for_all_nodes(
        self, node, clear_X_M=True
    ) -> None:
        if node is None:
            return
        node.missingness_reliance = self.compute_missingness_reliance(
            node.X, node.M, max_depth=node.depth
        )
        if clear_X_M:
            del node.X
            del node.M
        self._compute_missingness_reliance_for_all_nodes(
            node.left_subtree, clear_X_M
        )
        self._compute_missingness_reliance_for_all_nodes(
            node.right_subtree, clear_X_M
        )

    def compute_missingness_reliance(
        self,
        X,
        M,
        sample_mask=None,
        reduce=True,
        max_depth=None,
    ):
        check_is_fitted(self)
        miss_reliance = np.zeros_like(M)
        for i, (x, m) in enumerate(zip(X, M)):
            if sample_mask is not None and not sample_mask[i]:
                continue
            for feature in self._get_features_along_decision_path(
                x, max_depth
            ):
                miss_reliance[i, feature] = m[feature]
        if reduce:
            return np.mean(np.max(miss_reliance, axis=1))
        return miss_reliance

    def _get_split_candidates(
        self, X, y, M, sample_weight, depth, random_state
    ):
        del depth
        features = np.arange(self.n_features_in_)
        random_state.shuffle(features)
        for feature in features:
            yield self._best_split(
                X, y, feature, sample_weight=sample_weight, M=M
            )

    @abstractmethod
    def _get_node_value(self, y, sample_weight):
        """Get the value of a node in the decision tree."""

    @abstractmethod
    def _get_node_impurity(self, node_value, y, sample_weight):
        """Get the impurity of a node in the decision tree."""

    @abstractmethod
    def _is_homogeneous(self, node_value, y, sample_weight):
        """Determine whether a node is homogeneous."""

    @abstractmethod
    def _best_split(self, X, y, feature, sample_weight, M=None):
        """Find the best split for a given feature."""


class MALassoMixin:
    def _transform_input(self, X: np.ndarray, M: np.ndarray | None = None):
        if M is None:
            M = np.zeros_like(X)
        n_samples = X.shape[0]
        missingness = np.sum(M, axis=0)
        lambda_j = ((missingness + self.beta) / n_samples) * self.alpha
        return X * (self.alpha / lambda_j)

    def compute_missingness_reliance(self, X, M):
        check_is_fitted(self)
        return get_lm_missingness_reliance(self, X, M)
