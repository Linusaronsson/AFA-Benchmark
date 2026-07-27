"""
Measure legal static routes and per-method missingness effects.

The static analysis operates in the benchmark's *selection* space. A selection
may reveal several raw columns and has the same cost used during AFA evaluation.
The achieved-policy analysis never averages methods, mechanisms, rates, budgets,
or dataset instances before forming a contrast.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, final

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score

from afabench.components.unmaskers.config import UnmaskerConfig
from afabench.components.unmaskers.utils import get_afa_unmasker_from_config
from afabench.core.bundle_system.bundle import load_bundle

if TYPE_CHECKING:
    from afabench.core.types import AFADataset, AFAUnmasker

type Route = tuple[int, ...]

_NON_GREEDY_METHODS = ("aaco", "ol_without_mask")
_GREEDY_METHOD = "dime"
_F_SCORE_DATASETS = frozenset(
    {
        "actg",
        "bank_marketing",
        "ckd",
        "diabetes",
        "nhanes_mortality",
        "physionet",
    }
)
_CELL_KEYS = [
    "dataset",
    "method",
    "mechanism",
    "p",
    "strategy",
    "instance",
    "eval_hard_budget",
]


@final
class RouteScorer:
    """Cache classifier predictions for fixed legal selection sets."""

    def __init__(
        self,
        classifier: Any,  # noqa: ANN401
        features: torch.Tensor,
        labels: torch.Tensor,
        selection_feature_masks: torch.Tensor,
        feature_shape: torch.Size,
        metric: str,
        *,
        route_batch_size: int,
    ) -> None:
        self.classifier = classifier
        self.features = features
        self.labels = labels
        self.selection_feature_masks = selection_feature_masks
        self.feature_shape = feature_shape
        self.metric = metric
        self.route_batch_size = route_batch_size
        self._predictions: dict[Route, npt.NDArray[np.int64]] = {}
        self._true = labels.argmax(-1).detach().cpu().numpy().astype(np.int64)

    def _feature_mask(self, route: Route) -> torch.Tensor:
        if not route:
            return torch.zeros(
                self.feature_shape,
                dtype=torch.bool,
                device=self.features.device,
            )
        indices = torch.as_tensor(route, device=self.features.device)
        return self.selection_feature_masks[indices].any(dim=0)

    @torch.inference_mode()
    def predictions(self, routes: list[Route]) -> list[npt.NDArray[np.int64]]:
        missing = list(
            dict.fromkeys(r for r in routes if r not in self._predictions)
        )
        n = len(self.features)
        for start in range(0, len(missing), self.route_batch_size):
            batch = missing[start : start + self.route_batch_size]
            route_masks = torch.stack([self._feature_mask(r) for r in batch])
            masks = route_masks[:, None].expand(-1, n, *self.feature_shape)
            features = self.features[None].expand(
                len(batch), -1, *self.feature_shape
            )
            flat_masks = masks.reshape(-1, *self.feature_shape)
            flat_features = features.reshape(-1, *self.feature_shape)
            logits = self.classifier(
                flat_features * flat_masks,
                flat_masks,
                feature_shape=self.feature_shape,
            )
            predictions = (
                logits.argmax(-1).reshape(len(batch), n).cpu().numpy()
            )
            for route, prediction in zip(batch, predictions, strict=True):
                self._predictions[route] = prediction.astype(np.int64)
        return [self._predictions[r] for r in routes]

    def evaluate(
        self, routes: list[Route]
    ) -> tuple[
        npt.NDArray[np.bool_],
        npt.NDArray[np.float64],
    ]:
        predictions = self.predictions(routes)
        correct = np.stack(
            [prediction == self._true for prediction in predictions]
        )
        scores = (
            np.asarray(
                [
                    f1_score(
                        self._true,
                        prediction,
                        average="macro",
                        zero_division=cast("Any", 0),
                    )
                    for prediction in predictions
                ]
            )
            if self.metric == "f_score"
            else correct.mean(axis=1)
        )
        return correct, scores.astype(np.float64)


def primary_metric(dataset: str) -> str:
    return "f_score" if dataset in _F_SCORE_DATASETS else "accuracy"


def selection_feature_masks(
    unmasker: AFAUnmasker,
    feature_shape: torch.Size,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return the raw-feature mask revealed by each legal selection."""
    n_selections = unmasker.get_n_selections(feature_shape)
    masks = []
    for selection in range(n_selections):
        feature_mask = torch.zeros(
            (1, *feature_shape), dtype=torch.bool, device=device
        )
        selection_mask = torch.zeros(
            (1, n_selections), dtype=torch.bool, device=device
        )
        values = torch.zeros((1, *feature_shape), device=device)
        revealed = unmasker.unmask(
            values,
            feature_mask,
            values,
            torch.tensor([[selection]], device=device),
            selection_mask,
            feature_shape=feature_shape,
        )
        masks.append(revealed[0])
    output = torch.stack(masks)
    if not output.flatten(start_dim=1).any(dim=1).all():
        msg = "Every selection must reveal at least one feature"
        raise ValueError(msg)
    return output


def sample_feasible_routes(
    costs: npt.NDArray[np.float64],
    budget: float,
    k: int,
    generator: np.random.Generator,
) -> list[Route]:
    """Sample unique maximal routes without ever exceeding the cost budget."""
    routes: set[Route] = set()
    attempts = 0
    while len(routes) < k and attempts < max(100, 20 * k):
        attempts += 1
        spent = 0.0
        selected: list[int] = []
        for selection in generator.permutation(len(costs)):
            cost = float(costs[selection])
            if spent + cost <= budget + 1e-9:
                selected.append(int(selection))
                spent += cost
        routes.add(tuple(sorted(selected)))
    return sorted(routes)


def _route_cost(route: Route, costs: npt.NDArray[np.float64]) -> float:
    return float(costs[np.asarray(route, dtype=int)].sum()) if route else 0.0


def _greedy_route(
    scorer: RouteScorer,
    costs: npt.NDArray[np.float64],
    budget: float,
) -> Route:
    route: Route = ()
    current = float(scorer.evaluate([route])[1][0])
    while True:
        candidates = [
            tuple(sorted((*route, selection)))
            for selection in range(len(costs))
            if selection not in route
            and _route_cost((*route, selection), costs) <= budget + 1e-9
        ]
        if not candidates:
            return route
        scores = scorer.evaluate(candidates)[1]
        best = int(scores.argmax())
        if float(scores[best]) <= current + 1e-12:
            return route
        route, current = candidates[best], float(scores[best])


def _local_search(
    initial: Route,
    scorer: RouteScorer,
    costs: npt.NDArray[np.float64],
    budget: float,
) -> Route:
    route = initial
    current = float(scorer.evaluate([route])[1][0])
    while True:
        selected = set(route)
        unselected = set(range(len(costs))) - selected
        neighbors: set[Route] = {
            tuple(sorted(selected - {old})) for old in selected
        }
        neighbors |= {
            tuple(sorted(selected | {new}))
            for new in unselected
            if _route_cost(tuple(selected | {new}), costs) <= budget + 1e-9
        }
        neighbors |= {
            tuple(sorted((selected - {old}) | {new}))
            for old in selected
            for new in unselected
            if _route_cost(tuple((selected - {old}) | {new}), costs)
            <= budget + 1e-9
        }
        neighbors.discard(route)
        if not neighbors:
            return route
        candidates = sorted(neighbors)
        scores = scorer.evaluate(candidates)[1]
        best = int(scores.argmax())
        if float(scores[best]) <= current + 1e-12:
            return route
        route, current = candidates[best], float(scores[best])


def static_reference(
    sampled: list[Route],
    scorer: RouteScorer,
    costs: npt.NDArray[np.float64],
    budget: float,
) -> Route:
    """Best found route after random search, greedy search, and 1-swap refinement."""
    greedy = _greedy_route(scorer, costs, budget)
    candidates = list(dict.fromkeys([(), greedy, *sampled]))
    scores = scorer.evaluate(candidates)[1]
    start = candidates[int(scores.argmax())]
    return _local_search(start, scorer, costs, budget)


def route_metrics(
    correct: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    selection_scores: npt.NDArray[np.float64],
    top_frac: float,
) -> dict[str, float]:
    n_top = min(len(scores), max(2, int(np.ceil(top_frac * len(scores)))))
    ranking = np.argsort(selection_scores)[::-1]
    top = correct[ranking[:n_top]].astype(float)
    top = top[top.var(axis=1) > 0]
    correlation = float("nan")
    if len(top) >= 2:
        matrix = np.corrcoef(top)
        correlation = float(
            np.nanmean(matrix[np.triu_indices_from(matrix, 1)])
        )
    return {
        "random_route_score_mean": float(scores.mean()),
        "selected_sampled_route_score": float(scores[ranking[0]]),
        "top_route_correctness_correlation": correlation,
    }


def compute_effects(
    instance_metrics: pd.DataFrame,
    routes: pd.DataFrame,
    *,
    non_greedy_methods: tuple[str, ...] = _NON_GREEDY_METHODS,
    greedy_method: str = _GREEDY_METHOD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Form paired largest-budget contrasts without pooling cells."""
    duplicates = instance_metrics.duplicated(_CELL_KEYS, keep=False)
    if duplicates.any():
        msg = "Duplicate exact cells in instance_metrics"
        raise ValueError(msg)
    route_metrics_match = routes.apply(
        lambda row: row["metric"] == primary_metric(str(row["dataset"])),
        axis=1,
    )
    if not bool(route_metrics_match.all()):
        msg = "Route rows must use each dataset's primary metric"
        raise ValueError(msg)

    largest_budgets = {
        str(dataset): float(budget)
        for dataset, budget in routes.groupby("dataset")["budget"]
        .max()
        .items()
    }
    route_is_largest = np.asarray(
        [
            np.isclose(
                float(budget), largest_budgets.get(str(dataset), np.nan)
            )
            for dataset, budget in routes[["dataset", "budget"]].itertuples(
                index=False, name=None
            )
        ]
    )
    metric_is_largest = np.asarray(
        [
            np.isclose(
                float(budget), largest_budgets.get(str(dataset), np.nan)
            )
            for dataset, budget in instance_metrics[
                ["dataset", "eval_hard_budget"]
            ].itertuples(index=False, name=None)
        ]
    )
    routes = routes.loc[route_is_largest]
    instance_metrics = instance_metrics.loc[metric_is_largest]

    complete = instance_metrics.loc[
        (instance_metrics["mechanism"] == "none")
        & (instance_metrics["strategy"] == "complete")
    ].copy()
    complete["primary_score"] = complete.apply(
        lambda row: row[primary_metric(str(row["dataset"]))], axis=1
    )
    complete_key = ["dataset", "method", "instance", "eval_hard_budget"]
    complete_scores = complete.set_index(complete_key)["primary_score"]
    static_key = ["dataset", "instance", "budget"]
    if routes.duplicated(static_key).any():
        msg = "Duplicate static-route cells"
        raise ValueError(msg)
    static_scores = routes.set_index(static_key)["static_reference_score"]

    planning_rows: list[dict[str, Any]] = []
    for (dataset, method, instance, budget), score in complete_scores.items():
        if method not in non_greedy_methods:
            continue
        static = static_scores.get((dataset, instance, budget))
        greedy = complete_scores.get(
            (dataset, greedy_method, instance, budget)
        )
        if static is None or greedy is None:
            continue
        planning_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "instance": int(instance),
                "eval_hard_budget": float(budget),
                "metric": primary_metric(str(dataset)),
                "complete_score": float(score),
                "static_reference_score": float(static),
                "dime_complete_score": float(greedy),
                "adaptive_gain": float(score - static),
                "nongreedy_gain": float(score - greedy),
            }
        )

    missing = instance_metrics.loc[
        instance_metrics["strategy"].isin(
            ["restricted", "pvae_label_conditioned"]
        )
        & instance_metrics["method"].isin(non_greedy_methods)
    ].copy()
    missing["primary_score"] = missing.apply(
        lambda row: row[primary_metric(str(row["dataset"]))], axis=1
    )
    missing_key = [
        "dataset",
        "method",
        "instance",
        "mechanism",
        "p",
        "eval_hard_budget",
    ]
    restricted_scores = missing.loc[
        missing["strategy"] == "restricted"
    ].set_index(missing_key)["primary_score"]
    restored_scores = missing.loc[
        missing["strategy"] == "pvae_label_conditioned"
    ].set_index(missing_key)["primary_score"]
    missing_rows: list[dict[str, Any]] = []
    cells = restricted_scores.index.intersection(restored_scores.index)
    for dataset, method, instance, mechanism, probability, budget in sorted(
        cells
    ):
        reference = complete_scores.get((dataset, method, instance, budget))
        if reference is None:
            continue
        key = (dataset, method, instance, mechanism, probability, budget)
        restricted = float(restricted_scores.loc[key])
        restored = float(restored_scores.loc[key])
        missing_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "instance": int(instance),
                "mechanism": mechanism,
                "p": float(probability),
                "eval_hard_budget": float(budget),
                "metric": primary_metric(str(dataset)),
                "complete_score": float(reference),
                "restricted_score": restricted,
                "pvae_label_conditioned_score": restored,
                "missingness_damage": float(reference - restricted),
                "restoration_gain": restored - restricted,
            }
        )
    return pd.DataFrame(planning_rows), pd.DataFrame(missing_rows)


def gate_summary(
    planning: pd.DataFrame,
    missingness: pd.DataFrame,
    *,
    threshold: float = 0.01,
    expected_instances: int = 5,
    min_positive_instances: int = 4,
    mechanism: str = "mcar",
    probability: float = 0.7,
    methods: tuple[str, ...] = _NON_GREEDY_METHODS,
) -> pd.DataFrame:
    """Apply the predeclared gate independently to AACO and OL."""
    if missingness.empty:
        missingness = pd.DataFrame(
            columns=pd.Index(
                [
                    "dataset",
                    "method",
                    "eval_hard_budget",
                    "mechanism",
                    "p",
                    "restoration_gain",
                ]
            )
        )
    rows: list[dict[str, Any]] = []
    cells = planning[["dataset", "eval_hard_budget"]].drop_duplicates()
    for dataset, budget in cells.itertuples(index=False, name=None):
        dataset_rows: list[dict[str, Any]] = []
        for method in methods:
            plan = planning.loc[
                (planning["dataset"] == dataset)
                & (planning["method"] == method)
                & (planning["eval_hard_budget"] == budget)
            ]
            restore = missingness.loc[
                (missingness["dataset"] == dataset)
                & (missingness["method"] == method)
                & (missingness["eval_hard_budget"] == budget)
                & (missingness["mechanism"] == mechanism)
                & np.isclose(missingness["p"], probability)
            ]
            adaptive_mean = float(plan["adaptive_gain"].mean())
            nongreedy_mean = float(plan["nongreedy_gain"].mean())
            restoration_mean = float(restore["restoration_gain"].mean())
            planning_pass = bool(
                len(plan) == expected_instances
                and adaptive_mean >= threshold
                and nongreedy_mean >= threshold
                and int((plan["adaptive_gain"] > 0).sum())
                >= min_positive_instances
                and int((plan["nongreedy_gain"] > 0).sum())
                >= min_positive_instances
            )
            restoration_pass = bool(
                len(restore) == expected_instances
                and restoration_mean >= threshold
                and int((restore["restoration_gain"] > 0).sum())
                >= min_positive_instances
            )
            row = {
                "dataset": dataset,
                "method": method,
                "eval_hard_budget": float(budget),
                "n_planning_instances": len(plan),
                "adaptive_gain_mean": adaptive_mean,
                "nongreedy_gain_mean": nongreedy_mean,
                "planning_pass": planning_pass,
                "n_restoration_instances": len(restore),
                "restoration_mechanism": mechanism,
                "restoration_p": probability,
                "restoration_gain_mean": restoration_mean,
                "restoration_pass": restoration_pass,
                "method_pass": planning_pass and restoration_pass,
            }
            dataset_rows.append(row)
        concordant = bool(
            len(dataset_rows) == len(methods)
            and all(bool(row["method_pass"]) for row in dataset_rows)
        )
        for row in dataset_rows:
            row["dataset_concordant"] = concordant
            rows.append(row)
    return pd.DataFrame(rows)


def planning_gate_summary(
    planning: pd.DataFrame,
    *,
    threshold: float = 0.01,
    expected_instances: int = 5,
    min_positive_instances: int = 4,
    methods: tuple[str, ...] = _NON_GREEDY_METHODS,
) -> pd.DataFrame:
    """Gate complete-data planning before running a restoration matrix."""
    rows: list[dict[str, Any]] = []
    cells = planning[["dataset", "eval_hard_budget"]].drop_duplicates()
    for dataset, budget in cells.itertuples(index=False, name=None):
        dataset_rows: list[dict[str, Any]] = []
        for method in methods:
            method_rows = planning.loc[
                (planning["dataset"] == dataset)
                & (planning["method"] == method)
                & (planning["eval_hard_budget"] == budget)
            ]
            adaptive_mean = float(method_rows["adaptive_gain"].mean())
            nongreedy_mean = float(method_rows["nongreedy_gain"].mean())
            passed = bool(
                len(method_rows) == expected_instances
                and adaptive_mean >= threshold
                and nongreedy_mean >= threshold
                and int((method_rows["adaptive_gain"] > 0).sum())
                >= min_positive_instances
                and int((method_rows["nongreedy_gain"] > 0).sum())
                >= min_positive_instances
            )
            row = {
                "dataset": dataset,
                "method": method,
                "eval_hard_budget": float(budget),
                "n_instances": len(method_rows),
                "adaptive_gain_mean": adaptive_mean,
                "adaptive_positive_instances": int(
                    (method_rows["adaptive_gain"] > 0).sum()
                ),
                "nongreedy_gain_mean": nongreedy_mean,
                "nongreedy_positive_instances": int(
                    (method_rows["nongreedy_gain"] > 0).sum()
                ),
                "planning_pass": passed,
            }
            dataset_rows.append(row)
        concordant = bool(
            len(dataset_rows) == len(methods)
            and all(bool(row["planning_pass"]) for row in dataset_rows)
        )
        for row in dataset_rows:
            row["dataset_concordant"] = concordant
            rows.append(row)
    return pd.DataFrame(rows)


def _read_unmasker(
    dataset: str,
    mapping_path: Path,
    config_root: Path,
) -> AFAUnmasker:
    mapping = yaml.safe_load(mapping_path.read_text())["unmaskers"]
    name = mapping.get(dataset, mapping["default"])
    raw = yaml.safe_load((config_root / f"{name}.yaml").read_text())
    return get_afa_unmasker_from_config(UnmaskerConfig(**raw))


def _subsample(
    dataset: AFADataset,
    max_samples: int,
    generator: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    features, labels = dataset.get_all_data()
    if len(features) <= max_samples:
        return features, labels
    indices = torch.as_tensor(
        generator.choice(len(features), size=max_samples, replace=False)
    )
    return features[indices], labels[indices]


def _instances(root: Path, namespace: str, dataset: str) -> list[int]:
    folder = root / "datasets" / namespace / dataset
    return sorted(
        int(path.name) for path in folder.iterdir() if path.name.isdigit()
    )


def process_dataset(  # noqa: PLR0915
    classifier_location: Path,
    arguments: argparse.Namespace,
    budgets: dict[str, list[float]],
) -> list[dict[str, Any]]:
    if classifier_location.suffix == ".bundle":
        dataset = classifier_location.name[len("dataset-") : -len(".bundle")]
        shared_classifier, _ = load_bundle(
            classifier_location,
            device=torch.device(arguments.device),
        )
    else:
        dataset = classifier_location.name[len("dataset-") :]
        shared_classifier = None
    if arguments.datasets and dataset not in arguments.datasets:
        return []
    metric = primary_metric(dataset)
    unmasker = _read_unmasker(
        dataset, arguments.unmaskers, arguments.unmasker_config_root
    )
    rows: list[dict[str, Any]] = []
    instances = arguments.instances or _instances(
        arguments.root, arguments.namespace, dataset
    )
    for instance in instances:
        if shared_classifier is None:
            classifier_path = (
                classifier_location / f"instance-{instance}.bundle"
            )
            if not classifier_path.exists():
                print(f"skip {dataset}/{instance}: missing classifier bundle")
                continue
            classifier, _ = load_bundle(
                classifier_path,
                device=torch.device(arguments.device),
            )
        else:
            classifier = shared_classifier
        seed = arguments.seed + instance + zlib.crc32(dataset.encode("utf-8"))
        generator = np.random.default_rng(seed)
        selection_path = (
            arguments.root
            / "datasets"
            / arguments.namespace
            / dataset
            / str(instance)
            / f"{arguments.selection_split}.bundle"
        )
        eval_path = selection_path.with_name(f"{arguments.split}.bundle")
        if not selection_path.exists() or not eval_path.exists():
            print(f"skip {dataset}/{instance}: missing dataset bundle")
            continue
        selection_dataset = cast(
            "AFADataset",
            cast("object", load_bundle(selection_path)[0]),
        )
        selection_features, selection_labels = _subsample(
            selection_dataset, arguments.max_samples, generator
        )
        if arguments.selection_split == arguments.split:
            eval_dataset = selection_dataset
            eval_features, eval_labels = selection_features, selection_labels
        else:
            eval_dataset = cast(
                "AFADataset", cast("object", load_bundle(eval_path)[0])
            )
            eval_features, eval_labels = _subsample(
                eval_dataset, arguments.max_samples, generator
            )
        device = torch.device(arguments.device)
        feature_shape = selection_dataset.feature_shape
        masks = selection_feature_masks(unmasker, feature_shape, device=device)
        feature_costs = selection_dataset.get_feature_acquisition_costs().to(
            device
        )
        costs = (
            unmasker.get_selection_costs(feature_costs)
            .flatten()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if not np.isfinite(costs).all() or (costs <= 0).any():
            msg = f"Invalid selection costs for {dataset}"
            raise ValueError(msg)
        selection_scorer = RouteScorer(
            classifier,
            selection_features.to(device),
            selection_labels.to(device),
            masks,
            feature_shape,
            metric,
            route_batch_size=arguments.route_batch_size,
        )
        eval_scorer = (
            selection_scorer
            if arguments.selection_split == arguments.split
            else RouteScorer(
                classifier,
                eval_features.to(device),
                eval_labels.to(device),
                masks,
                eval_dataset.feature_shape,
                metric,
                route_batch_size=arguments.route_batch_size,
            )
        )
        dataset_budgets = budgets.get(dataset, budgets["default"])
        for budget in dataset_budgets:
            sampled = sample_feasible_routes(
                costs, float(budget), arguments.k, generator
            )
            reference = static_reference(
                sampled, selection_scorer, costs, float(budget)
            )
            _, selection_scores = selection_scorer.evaluate(sampled)
            correct, scores = eval_scorer.evaluate(sampled)
            _, reference_score = eval_scorer.evaluate([reference])
            metrics = route_metrics(
                correct, scores, selection_scores, arguments.top_frac
            )
            rows.append(
                {
                    "namespace": arguments.namespace,
                    "dataset": dataset,
                    "instance": instance,
                    "selection_split": arguments.selection_split,
                    "eval_split": arguments.split,
                    "metric": metric,
                    "n_features": int(feature_shape.numel()),
                    "n_selections": len(costs),
                    "n_samples": len(eval_features),
                    "budget": float(budget),
                    "k_requested": arguments.k,
                    "k_unique": len(sampled),
                    "static_reference_route": json.dumps(reference),
                    "static_reference_cost": _route_cost(reference, costs),
                    "static_reference_score": float(reference_score[0]),
                    "route_sensitivity": float(
                        reference_score[0] - scores.mean()
                    ),
                    **metrics,
                }
            )
            print(
                f"{dataset:12s} i={instance} b={budget:g} "
                f"static={reference_score[0]:.3f} "
                f"sampled={metrics['selected_sampled_route_score']:.3f} "
                f"corr={metrics['top_route_correctness_correlation']:.3f}"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("extra/output/missing_data")
    )
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--selection-split", default="val")
    parser.add_argument(
        "--budgets",
        type=Path,
        default=Path("extra/workflow/conf/eval_hard_budgets/all.yaml"),
    )
    parser.add_argument(
        "--unmaskers",
        type=Path,
        default=Path("extra/workflow/conf/unmaskers/all.yaml"),
    )
    parser.add_argument(
        "--unmasker-config-root",
        type=Path,
        default=Path("extra/conf/components/unmaskers"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--planning-output", type=Path)
    parser.add_argument("--missingness-output", type=Path)
    parser.add_argument("--gate-output", type=Path)
    parser.add_argument("--planning-gate-output", type=Path)
    parser.add_argument("--datasets", nargs="*")
    parser.add_argument("--instances", nargs="*", type=int)
    parser.add_argument("--k", type=int, default=500)
    parser.add_argument("--top-frac", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--route-batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    budgets = yaml.safe_load(arguments.budgets.read_text())[
        "eval_hard_budgets"
    ]
    classifier_root = arguments.root / "classifier" / arguments.namespace
    classifier_locations = sorted(
        (arguments.root / "classifier" / arguments.namespace).glob(
            "dataset-*.bundle"
        )
    )
    classifier_locations.extend(
        sorted(
            path
            for path in classifier_root.glob("dataset-*")
            if path.is_dir() and path.suffix != ".bundle"
        )
    )
    rows = [
        row
        for path in classifier_locations
        for row in process_dataset(path, arguments, budgets)
    ]
    if not rows:
        msg = f"No route rows for namespace {arguments.namespace}"
        raise SystemExit(msg)
    analysis = arguments.root / "analysis"
    route_output = arguments.output or analysis / (
        f"route_redundancy_{arguments.namespace}.csv"
    )
    route_output.parent.mkdir(parents=True, exist_ok=True)
    routes = pd.DataFrame(rows)
    routes.to_csv(route_output, index=False)
    print(f"wrote {route_output}")

    instance_path = (
        arguments.root
        / "summary"
        / arguments.split
        / arguments.namespace
        / "instance_metrics.csv"
    )
    if not instance_path.exists():
        return
    planning, missingness = compute_effects(pd.read_csv(instance_path), routes)
    planning_output = arguments.planning_output or analysis / (
        f"planning_effects_{arguments.namespace}.csv"
    )
    missingness_output = arguments.missingness_output or analysis / (
        f"missingness_effects_{arguments.namespace}.csv"
    )
    gate_output = arguments.gate_output or analysis / (
        f"route_gate_{arguments.namespace}.csv"
    )
    planning_gate_output = arguments.planning_gate_output or analysis / (
        f"planning_gate_{arguments.namespace}.csv"
    )
    planning.to_csv(planning_output, index=False)
    missingness.to_csv(missingness_output, index=False)
    planning_gate_summary(planning).to_csv(
        planning_gate_output,
        index=False,
    )
    gate_summary(planning, missingness).to_csv(gate_output, index=False)
    print(
        "wrote "
        f"{planning_output}, {missingness_output}, "
        f"{planning_gate_output}, {gate_output}"
    )


if __name__ == "__main__":
    main()
