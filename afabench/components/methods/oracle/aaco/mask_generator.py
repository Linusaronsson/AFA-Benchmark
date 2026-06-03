from collections.abc import Iterable
from itertools import chain, combinations

import numpy as np
import numpy.typing as npt
import torch


def powerset[T](iterable: Iterable[T]) -> list[list[T]]:
    """
    Generate all possible subsets (powerset) of the iterable, excluding the empty set.

    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3).
    """
    s = list(iterable)
    return [
        list(x)
        for x in list(
            chain.from_iterable(
                combinations(s, r) for r in range(1, len(s) + 1)
            )
        )
    ]


def generate_all_masks(input_dim: int) -> npt.NDArray[np.float64]:
    """Generate all possible masks for a given feature dimension."""
    subsets = powerset(range(input_dim))
    all_masks = np.zeros((len(subsets), input_dim))

    for i in range(len(subsets)):
        all_masks[i, subsets[i]] = 1

    return all_masks


class AllMaskGenerator:
    def __init__(self, all_masks: npt.NDArray[np.float64]) -> None:
        self.all_masks: torch.Tensor = torch.from_numpy(all_masks)

    def __call__(self, _mask_curr: torch.Tensor) -> torch.Tensor:
        return self.all_masks


def generate_ball(
    num_masks: int,
    d1: int,
    d2: int,
    rng: np.random.Generator | None = None,
) -> npt.NDArray[np.float64]:
    """Generate random binary masks."""
    rng = rng or np.random.default_rng()
    ball = np.concatenate(
        [
            np.sum(
                rng.permutation(np.eye(d1))[:, : rng.integers(d2)],
                1,
                keepdims=True,
            )
            for _ in range(num_masks)
        ],
        1,
    )
    return ball


class RandomMaskGenerator:
    """Their exact random mask generator implementation."""

    def __init__(
        self, num_samples: int, feature_dim: int, num_generated_masks: int
    ) -> None:
        self.num_samples: int = num_samples
        self.feature_dim: int = feature_dim
        self.num_generated_masks: int = num_generated_masks
        self._cached_masks: torch.Tensor | None = None
        self._rng: np.random.Generator = np.random.default_rng()

    def __call__(self, _mask_curr: torch.Tensor) -> torch.Tensor:
        # Cache masks to avoid repeated numpy generation in tight loops.
        if self._cached_masks is None:
            ball = generate_ball(
                self.num_generated_masks,
                self.feature_dim,
                self.feature_dim,
                self._rng,
            )
            self._cached_masks = torch.tensor(
                ball[
                    :,
                    self._rng.permutation(self.num_generated_masks)[
                        : self.num_generated_masks
                    ],
                ],
                dtype=torch.float32,
            ).T
        return self._cached_masks


all_mask_generator = AllMaskGenerator
random_mask_generator = RandomMaskGenerator
