from itertools import chain, combinations

import numpy as np
import torch


def powerset(iterable):
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


def generate_all_masks(input_dim):
    """Generate all possible masks for a given feature dimension."""
    subsets = powerset(range(input_dim))
    all_masks = np.zeros((len(subsets), input_dim))

    for i in range(len(subsets)):
        all_masks[i, subsets[i]] = 1

    return all_masks


class all_mask_generator:
    def __init__(self, all_masks):
        self.all_masks = torch.from_numpy(all_masks)

    def __call__(self, mask_curr):
        return self.all_masks


def generate_ball(N, d1, d2):
    """Generate random binary masks."""
    Ball = np.concatenate(
        [
            np.sum(
                np.random.permutation(np.eye(d1))[:, : np.random.randint(d2)],
                1,
                keepdims=True,
            )
            for _ in range(N)
        ],
        1,
    )
    return Ball


class random_mask_generator:
    """Their exact random mask generator implementation."""

    def __init__(self, num_samples, feature_dim, num_generated_masks):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.num_generated_masks = num_generated_masks
        self._cached_masks = None

    def __call__(self, mask_curr):
        # Cache masks to avoid repeated numpy generation in tight loops.
        if self._cached_masks is None:
            ball = generate_ball(
                self.num_generated_masks, self.feature_dim, self.feature_dim
            )
            self._cached_masks = torch.tensor(
                ball[
                    :,
                    np.random.permutation(self.num_generated_masks)[
                        : self.num_generated_masks
                    ],
                ],
                dtype=torch.float32,
            ).T
        return self._cached_masks
