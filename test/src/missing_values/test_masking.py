import numpy as np
import pytest
import torch

from afabench.missing_values.masking import (
    MAR_mask,
    MNAR_mask_logistic,
    MNAR_mask_quantiles,
    MNAR_self_mask_logistic,
)


@pytest.mark.parametrize("as_torch", [False, True])
def test_mar_mask_single_sample_no_nan_or_crash(as_torch: bool) -> None:
    x_np = np.array([[0.2, -1.0, 0.3, 0.7, -0.5]], dtype=np.float32)
    x = torch.from_numpy(x_np) if as_torch else x_np

    mask = MAR_mask(x=x, p=0.3, p_obs=0.4, seed=1)

    if as_torch:
        assert isinstance(mask, torch.Tensor)
        assert mask.dtype == torch.bool
    else:
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.bool_
    assert mask.shape == x_np.shape


@pytest.mark.parametrize("as_torch", [False, True])
def test_mnar_masks_single_sample_no_crash(as_torch: bool) -> None:
    x_np = np.array([[0.5, -0.8, 0.1, 0.9, -0.2]], dtype=np.float32)
    x = torch.from_numpy(x_np) if as_torch else x_np

    masks = [
        MNAR_mask_logistic(x=x, p=0.3, p_params=0.4, seed=2),
        MNAR_self_mask_logistic(x=x, p=0.3, seed=3),
        MNAR_mask_quantiles(
            x=x,
            p=0.3,
            q=0.25,
            p_params=0.4,
            cut="both",
            seed=4,
        ),
    ]

    for mask in masks:
        if as_torch:
            assert isinstance(mask, torch.Tensor)
            assert mask.dtype == torch.bool
        else:
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.bool_
        assert mask.shape == x_np.shape
