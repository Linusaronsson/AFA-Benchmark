"""Run the exact finite-state control study used in the paper."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, final, overload

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

BUDGET = 2
SHORTCUT_ACCURACY = 0.75
LAPLACE = 0.5
CONTEXT, BLOCK0, BLOCK1, SHORTCUT = 0, 1, 2, 3

ARM_LOCAL = "mask_local"
ARM_AGNOSTIC = "mask_agnostic"
ARM_GENERATIVE = "generative"
ARM_COMPLETE = "complete"
ARMS = (ARM_LOCAL, ARM_AGNOSTIC, ARM_GENERATIVE, ARM_COMPLETE)

DIMENSIONS = (6, 8, 10)
MISSING_RATES = (0.3, 0.5, 0.7)
SAMPLE_SIZES = (
    10,
    18,
    32,
    56,
    100,
    178,
    316,
    562,
    1000,
    1778,
    3162,
    5623,
    10000,
    17783,
    31623,
    56234,
    100000,
)
DEFAULT_REPS = 150
DEFAULT_SEED = 0

UIntArray = npt.NDArray[np.uint8]
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.integer]


class StudyResult(NamedTuple):
    d: int
    p_miss: float
    n: int
    arm: str
    rep: int
    regret: float


@dataclass(frozen=True)
class Dataset:
    x: UIntArray
    y: UIntArray
    available: BoolArray


@dataclass(frozen=True)
class CountTables:
    root_count: float
    root_y1: float
    single_count: FloatArray
    single_y1: FloatArray
    pair_count: FloatArray
    pair_y1: FloatArray


@dataclass(frozen=True)
class Plan:
    root_action: int
    single_action: IntArray
    root_prediction: int
    single_prediction: UIntArray
    pair_prediction: UIntArray


class FixedPredictor(NamedTuple):
    root: int
    single: UIntArray
    pair: UIntArray

    def predict(
        self, features: tuple[int, ...], values: tuple[int, ...]
    ) -> int:
        if not features:
            return self.root
        if len(features) == 1:
            return int(self.single[features[0], values[0]])
        return int(self.pair[features[0], features[1], values[0], values[1]])


class PlanQuantities(NamedTuple):
    root_stop: float
    single_stop: FloatArray
    q_last: FloatArray


@final
class Problem:
    """The shortcut problem with any number of independent noise features."""

    def __init__(self, d: int):
        if d < 4:
            message = "The shortcut problem requires d >= 4."
            raise ValueError(message)
        self.d = d
        self.costs = np.ones(d, dtype=np.int8)
        self.costs[SHORTCUT] = BUDGET

        core = np.array(
            [[(code >> bit) & 1 for bit in range(4)] for code in range(16)],
            dtype=np.uint8,
        )
        labels = np.where(
            core[:, CONTEXT] == 0, core[:, BLOCK0], core[:, BLOCK1]
        )
        weights = np.where(
            core[:, SHORTCUT] == labels,
            SHORTCUT_ACCURACY / 8,
            (1 - SHORTCUT_ACCURACY) / 8,
        )
        if not np.isclose(weights.sum(), 1):
            message = "The core distribution must sum to one."
            raise AssertionError(message)
        self.core = core
        self.labels = labels.astype(np.uint8)
        self.weights = weights

    def sample_complete(
        self, rng: np.random.Generator, n: int
    ) -> tuple[UIntArray, UIntArray]:
        x = rng.integers(0, 2, size=(n, self.d), dtype=np.uint8)
        if n == 0:
            return x, np.empty(0, dtype=np.uint8)
        y = np.where(x[:, CONTEXT] == 0, x[:, BLOCK0], x[:, BLOCK1]).astype(
            np.uint8
        )
        faithful = rng.random(n) < SHORTCUT_ACCURACY
        x[:, SHORTCUT] = np.where(faithful, y, 1 - y)
        return x, y

    def sample(
        self, rng: np.random.Generator, n: int, p_miss: float
    ) -> Dataset:
        if not 0 <= p_miss < 1:
            message = "p_miss must lie in [0, 1)."
            raise ValueError(message)
        x, y = self.sample_complete(rng, n)
        available = rng.random((n, self.d)) >= p_miss
        return Dataset(x=x, y=y, available=available)

    def _posterior(
        self, features: tuple[int, ...], values: tuple[int, ...]
    ) -> FloatArray:
        keep = np.ones(len(self.core), dtype=bool)
        for feature, value in zip(features, values, strict=True):
            if feature < 4:
                keep &= self.core[:, feature] == value
        posterior = self.weights * keep
        total = posterior.sum()
        if total <= 0:
            raise AssertionError((features, values))
        return posterior / total

    def label_probability(
        self, features: tuple[int, ...], values: tuple[int, ...]
    ) -> float:
        return float(self._posterior(features, values) @ self.labels)

    def feature_probability(
        self,
        features: tuple[int, ...],
        values: tuple[int, ...],
        action: int,
    ) -> float:
        if action >= 4:
            return 0.5
        return float(self._posterior(features, values) @ self.core[:, action])

    def fixed_predictor(self) -> FixedPredictor:
        """Return the Bayes predictor shared by every training view."""
        root = int(self.label_probability((), ()) >= 0.5)
        single = np.empty((self.d, 2), dtype=np.uint8)
        pair = np.empty((self.d, self.d, 2, 2), dtype=np.uint8)
        for first in range(self.d):
            for first_value in (0, 1):
                single[first, first_value] = int(
                    self.label_probability((first,), (first_value,)) >= 0.5
                )
                for second in range(self.d):
                    for second_value in (0, 1):
                        if second == first:
                            pair[first, second, first_value, second_value] = (
                                single[first, first_value]
                            )
                            continue
                        pair[first, second, first_value, second_value] = int(
                            self.label_probability(
                                (first, second),
                                (first_value, second_value),
                            )
                            >= 0.5
                        )
        return FixedPredictor(root, single, pair)

    def optimal_value(self) -> float:
        return self._optimal_value((), (), BUDGET, self.fixed_predictor())

    def _optimal_value(
        self,
        features: tuple[int, ...],
        values: tuple[int, ...],
        remaining_budget: int,
        predictor: FixedPredictor,
    ) -> float:
        p_y1 = self.label_probability(features, values)
        prediction = predictor.predict(features, values)
        best = p_y1 if prediction == 1 else 1 - p_y1
        for action, cost in enumerate(self.costs):
            if action in features or cost > remaining_budget:
                continue
            p_one = self.feature_probability(features, values, action)
            next_budget = remaining_budget - int(cost)
            candidate = (1 - p_one) * self._optimal_value(
                (*features, action), (*values, 0), next_budget, predictor
            ) + p_one * self._optimal_value(
                (*features, action), (*values, 1), next_budget, predictor
            )
            best = max(best, candidate)
        return best

    def evaluate(self, plan: Plan) -> float:
        def recurse(
            features: tuple[int, ...],
            values: tuple[int, ...],
            remaining_budget: int,
        ) -> float:
            if not features:
                action = plan.root_action
                prediction = plan.root_prediction
            elif len(features) == 1:
                action = int(plan.single_action[features[0], values[0]])
                prediction = int(
                    plan.single_prediction[features[0], values[0]]
                )
            else:
                action = -1
                prediction = int(
                    plan.pair_prediction[
                        features[0], features[1], values[0], values[1]
                    ]
                )

            if action < 0:
                p_y1 = self.label_probability(features, values)
                return p_y1 if prediction == 1 else 1 - p_y1
            if action in features or self.costs[action] > remaining_budget:
                message = "Plan contains an illegal acquisition."
                raise AssertionError(message)

            p_one = self.feature_probability(features, values, action)
            next_budget = remaining_budget - int(self.costs[action])
            return (1 - p_one) * recurse(
                (*features, action), (*values, 0), next_budget
            ) + p_one * recurse((*features, action), (*values, 1), next_budget)

        return recurse((), (), BUDGET)


@overload
def _smoothed_probability(successes: float, total: float) -> float: ...


@overload
def _smoothed_probability(
    successes: FloatArray, total: FloatArray
) -> FloatArray: ...


def _smoothed_probability(
    successes: FloatArray | float, total: FloatArray | float
) -> FloatArray | float:
    return (successes + LAPLACE) / (total + 2 * LAPLACE)


def build_count_tables(
    data: Dataset, *, complete_only: bool = False
) -> CountTables:
    if complete_only:
        selected = data.available.all(axis=1)
        x, y = data.x[selected], data.y[selected]
        available = np.ones_like(x, dtype=bool)
    else:
        x, y, available = data.x, data.y, data.available

    n, d = x.shape
    if n == 0:
        zeros_single = np.zeros((d, 2), dtype=np.float64)
        zeros_pair = np.zeros((d, d, 2, 2), dtype=np.float64)
        return CountTables(
            0.0,
            0.0,
            zeros_single,
            zeros_single.copy(),
            zeros_pair,
            zeros_pair.copy(),
        )

    z0 = (available & (x == 0)).astype(np.float64)
    z1 = (available & (x == 1)).astype(np.float64)
    yf = y.astype(np.float64)
    z0y, z1y = z0 * yf[:, None], z1 * yf[:, None]

    single_count = np.stack((z0.sum(axis=0), z1.sum(axis=0)), axis=1)
    single_y1 = np.stack((z0y.sum(axis=0), z1y.sum(axis=0)), axis=1)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        count00, count01, count11 = z0.T @ z0, z0.T @ z1, z1.T @ z1
        y00, y01, y11 = z0.T @ z0y, z0.T @ z1y, z1.T @ z1y
    tables = (count00, count01, count11, y00, y01, y11)
    if not all(np.isfinite(table).all() for table in tables):
        message = "Non-finite sufficient statistics."
        raise FloatingPointError(message)

    pair_count = np.empty((d, d, 2, 2), dtype=np.float64)
    pair_y1 = np.empty_like(pair_count)
    pair_count[:, :, 0, 0], pair_count[:, :, 0, 1] = count00, count01
    pair_count[:, :, 1, 0], pair_count[:, :, 1, 1] = count01.T, count11
    pair_y1[:, :, 0, 0], pair_y1[:, :, 0, 1] = y00, y01
    pair_y1[:, :, 1, 0], pair_y1[:, :, 1, 1] = y01.T, y11
    return CountTables(
        float(n),
        float(y.sum()),
        single_count,
        single_y1,
        pair_count,
        pair_y1,
    )


def _base_plan_quantities(
    tables: CountTables, costs: IntArray, predictor: FixedPredictor
) -> PlanQuantities:
    d = len(tables.single_count)
    root_p1 = float(_smoothed_probability(tables.root_y1, tables.root_count))
    single_p1 = _smoothed_probability(tables.single_y1, tables.single_count)
    pair_p1 = _smoothed_probability(tables.pair_y1, tables.pair_count)
    root_stop = root_p1 if predictor.root == 1 else 1 - root_p1
    single_stop = np.where(predictor.single == 1, single_p1, 1 - single_p1)
    pair_stop = np.where(predictor.pair == 1, pair_p1, 1 - pair_p1)

    q_last = np.full((d, 2, d), -np.inf, dtype=np.float64)
    for first in range(d):
        for action in range(d):
            if action == first or costs[first] + costs[action] > BUDGET:
                continue
            for first_value in (0, 1):
                counts = tables.pair_count[first, action, first_value]
                p_one = float(_smoothed_probability(counts[1], counts.sum()))
                stops = pair_stop[first, action, first_value]
                q_last[first, first_value, action] = (1 - p_one) * stops[
                    0
                ] + p_one * stops[1]
    return PlanQuantities(root_stop, single_stop, q_last)


def _assemble_plan(
    root_q: FloatArray,
    quantities: PlanQuantities,
    predictor: FixedPredictor,
) -> Plan:
    root_action = -1
    root_best = quantities.root_stop
    for action, value in enumerate(root_q):
        if value > root_best + 1e-12:
            root_best, root_action = float(value), action

    d = len(root_q)
    single_action = np.full((d, 2), -1, dtype=np.int16)
    for first in range(d):
        for value in (0, 1):
            best = quantities.single_stop[first, value]
            for action in range(d):
                if quantities.q_last[first, value, action] > best + 1e-12:
                    best = quantities.q_last[first, value, action]
                    single_action[first, value] = action

    return Plan(
        root_action,
        single_action,
        predictor.root,
        predictor.single,
        predictor.pair,
    )


def plan_model_based(
    tables: CountTables, costs: IntArray, predictor: FixedPredictor
) -> Plan:
    quantities = _base_plan_quantities(tables, costs, predictor)
    root_q = np.full(len(costs), -np.inf, dtype=np.float64)
    for action, cost in enumerate(costs):
        if cost > BUDGET:
            continue
        counts = tables.single_count[action]
        p_one = float(_smoothed_probability(counts[1], counts.sum()))
        values = quantities.single_stop[action].copy()
        if cost < BUDGET:
            values = np.maximum(values, quantities.q_last[action].max(axis=1))
        root_q[action] = (1 - p_one) * values[0] + p_one * values[1]
    return _assemble_plan(root_q, quantities, predictor)


def plan_mask_agnostic(
    data: Dataset,
    tables: CountTables,
    costs: IntArray,
    predictor: FixedPredictor,
) -> Plan:
    """Fit one shared Q table using each instance's legal continuation set."""
    quantities = _base_plan_quantities(tables, costs, predictor)
    root_q = np.full(len(costs), -np.inf, dtype=np.float64)
    for action, cost in enumerate(costs):
        if cost > BUDGET:
            continue
        selected = data.available[:, action]
        count = int(selected.sum())
        if count == 0:
            root_q[action] = 0.5
            continue
        values = quantities.single_stop[action, data.x[:, action]].copy()
        for continuation, continuation_cost in enumerate(costs):
            if cost + continuation_cost > BUDGET:
                continue
            candidate = quantities.q_last[
                action, data.x[:, action], continuation
            ]
            values = np.where(
                data.available[:, continuation],
                np.maximum(values, candidate),
                values,
            )
        root_q[action] = (float(values[selected].sum()) + LAPLACE) / (
            count + 2 * LAPLACE
        )
    return _assemble_plan(root_q, quantities, predictor)


def fit_regrets(problem: Problem, data: Dataset) -> dict[str, float]:
    pooled = build_count_tables(data)
    complete_only = build_count_tables(data, complete_only=True)
    predictor = problem.fixed_predictor()
    complete_data = Dataset(
        data.x, data.y, np.ones_like(data.available, dtype=bool)
    )
    plans = {
        ARM_LOCAL: plan_model_based(complete_only, problem.costs, predictor),
        ARM_AGNOSTIC: plan_mask_agnostic(
            data, pooled, problem.costs, predictor
        ),
        ARM_GENERATIVE: plan_model_based(pooled, problem.costs, predictor),
        ARM_COMPLETE: plan_model_based(
            build_count_tables(complete_data), problem.costs, predictor
        ),
    }
    optimum = problem.optimal_value()
    return {
        arm: max(0.0, optimum - problem.evaluate(plan))
        for arm, plan in plans.items()
    }


def _rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence((seed, *keys)))


def _study_task(args: tuple[int, float, int, int, int]) -> list[StudyResult]:
    d, p_miss, n, rep, seed = args
    problem = Problem(d)
    rng = _rng(seed, d, round(1000 * p_miss), n, rep)
    regrets = fit_regrets(problem, problem.sample(rng, n, p_miss))
    return [StudyResult(d, p_miss, n, arm, rep, regrets[arm]) for arm in ARMS]


def _map_tasks[TaskT, ResultT](
    function: Callable[[TaskT], ResultT], tasks: Sequence[TaskT], jobs: int
) -> list[ResultT]:
    if jobs == 1:
        return [function(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(
            executor.map(
                function, tasks, chunksize=max(1, len(tasks) // (jobs * 8))
            )
        )


def run_study(
    *, reps: int, seed: int, jobs: int, smoke: bool = False
) -> list[StudyResult]:
    dimensions = (6,) if smoke else DIMENSIONS
    sample_sizes = (10, 100, 1000) if smoke else SAMPLE_SIZES
    run_reps = min(reps, 3) if smoke else reps
    tasks = [
        (d, p_miss, n, rep, seed)
        for d in dimensions
        for p_miss in MISSING_RATES
        for n in sample_sizes
        for rep in range(run_reps)
    ]
    nested = _map_tasks(_study_task, tasks, jobs)
    return [result for results in nested for result in results]


def write_results(path: Path, results: Iterable[StudyResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(StudyResult._fields)
        writer.writerows(results)


def write_log(path: Path, results: Sequence[StudyResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[int, float, int], list[StudyResult]] = {}
    for result in results:
        grouped.setdefault((result.d, result.p_miss, result.n), []).append(
            result
        )
    lines = []
    for (d, p_miss, n), group in grouped.items():
        means = {
            arm: np.mean([item.regret for item in group if item.arm == arm])
            for arm in ARMS
        }
        summary = " ".join(f"{arm}={means[arm]:.4f}" for arm in ARMS)
        lines.append(f"d={d} p={p_miss:g} n={n:6d} regret {summary}")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--jobs", type=int, default=min(os.cpu_count() or 1, 8)
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extra/output/paper/experiments/results"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reps <= 0 or args.jobs <= 0:
        message = "reps and jobs must be positive."
        raise ValueError(message)
    results = run_study(
        reps=args.reps, seed=args.seed, jobs=args.jobs, smoke=args.smoke
    )
    write_results(args.output_dir / "exact_study.csv", results)
    write_log(args.output_dir / "exact_study.log", results)
    print(f"wrote {len(results):,} results to {args.output_dir}")


if __name__ == "__main__":
    main()
