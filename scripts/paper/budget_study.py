"""
Run the budget sweep with paired myopic/full-horizon policies.

The original b=2 experiment is reused verbatim. For b=3,4, a context selects
one of two parity branches of b-1 bits; the noisy shortcut costs b. Every
training set also supplies an unmasked complete-data control.
"""

from __future__ import annotations

import argparse
import csv
import os
from functools import cache
from pathlib import Path
from typing import NamedTuple

from scripts.paper.budget_problem import BudgetProblem
from scripts.paper.exact_study import (
    ARMS,
    DEFAULT_REPS,
    DEFAULT_SEED,
    MISSING_RATES,
    SAMPLE_SIZES,
    _map_tasks,
    _rng,
    _study_task,
)

BUDGETS = (2, 3, 4)
DIMENSION = 10


class BudgetResult(NamedTuple):
    d: int
    p_miss: float
    n: int
    arm: str
    rep: int
    regret: float
    horizon: int
    population_accuracy: float
    budget: int


@cache
def problem_for(budget: int) -> BudgetProblem:
    return BudgetProblem(DIMENSION, budget)


@cache
def population_accuracies(budget: int) -> dict[int, float]:
    problem = problem_for(budget)
    return {
        horizon: problem.evaluate(policy)
        for horizon, policy in problem.model_plans(
            problem.truth_stop, problem.truth_transition
        ).items()
    }


def study_task(args: tuple[int, float, int, int, int]) -> list[BudgetResult]:
    budget, p_miss, n, rep, seed = args
    if budget == 2:
        original = _study_task((DIMENSION, p_miss, n, rep, seed))
        return [BudgetResult(*row, budget) for row in original.regrets]
    problem = problem_for(budget)
    rng = _rng(seed, DIMENSION, round(1000 * p_miss), n, rep, budget)
    plans = problem.fit(problem.sample(rng, n, p_miss))
    population = population_accuracies(budget)
    return [
        BudgetResult(
            DIMENSION,
            p_miss,
            n,
            arm,
            rep,
            max(
                0.0, population[budget] - problem.evaluate(plans[arm, horizon])
            ),
            horizon,
            population[horizon],
            budget,
        )
        for arm in ARMS
        for horizon in (1, budget)
    ]


def run_study(
    *, reps: int, seed: int, jobs: int, smoke: bool = False
) -> list[BudgetResult]:
    if reps <= 0 or jobs <= 0:
        message = "reps and jobs must be positive."
        raise ValueError(message)
    sizes = (10, 100, 1000) if smoke else SAMPLE_SIZES
    tasks = [
        (budget, p, n, rep, seed)
        for budget in BUDGETS
        for p in MISSING_RATES
        for n in sizes
        for rep in range(min(reps, 3) if smoke else reps)
    ]
    return [
        row for rows in _map_tasks(study_task, tasks, jobs) for row in rows
    ]


def write_results(path: Path, results: list[BudgetResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(BudgetResult._fields)
        writer.writerows(results)


def main() -> None:
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
    args = parser.parse_args()
    results = run_study(
        reps=args.reps, seed=args.seed, jobs=args.jobs, smoke=args.smoke
    )
    write_results(args.output_dir / "budget_study.csv", results)
    print(f"wrote {len(results):,} regrets to {args.output_dir}")


if __name__ == "__main__":
    main()
