import torch

from afabench.components.methods.oracle.aaco.core import AACOOracle
from afabench.components.methods.oracle.aaco.mask_generator import (
    random_mask_generator,
)

_MASK_CURR = torch.zeros(8)


def test_same_seed_gives_identical_candidate_masks() -> None:
    a = random_mask_generator(100, 8, 100, 0)(_MASK_CURR)
    b = random_mask_generator(100, 8, 100, 0)(_MASK_CURR)
    assert torch.equal(a, b)


def test_different_seeds_give_different_candidate_masks() -> None:
    a = random_mask_generator(100, 8, 100, 0)(_MASK_CURR)
    b = random_mask_generator(100, 8, 100, 1)(_MASK_CURR)
    assert not torch.equal(a, b)


def test_fitted_oracles_share_masks_iff_they_share_a_seed() -> None:
    """The property the ladder depends on: same seed, same candidate masks."""
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def masks_for(mask_seed: int) -> torch.Tensor:
        oracle = AACOOracle(k_neighbors=1, mask_seed=mask_seed)
        oracle.fit(features, labels)
        assert oracle.mask_generator is not None
        return oracle.mask_generator(_MASK_CURR)

    assert torch.equal(masks_for(0), masks_for(0))
    assert not torch.equal(masks_for(0), masks_for(7))
