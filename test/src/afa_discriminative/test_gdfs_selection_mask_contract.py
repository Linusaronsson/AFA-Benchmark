import pytest
import torch
from torch import nn

from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
)


def _make_method(
    d_in: int, n_selections: int, d_out: int = 2
) -> GDFSAFAMethod:
    selector = nn.Linear(2 * d_in, n_selections)
    predictor = nn.Linear(2 * d_in, d_out)
    return GDFSAFAMethod(
        selector=selector,
        predictor=predictor,
        device=torch.device("cpu"),
        modality="tabular",
        d_in=d_in,
        d_out=d_out,
        n_selections=n_selections,
    )


def test_gdfs_grouped_selection_excludes_already_selected() -> None:
    """Context Unmasker-style grouping: n_selections != n_features."""
    d_in = 10
    n_contexts = 3
    n_selections = 1 + (d_in - n_contexts)  # 8
    method = _make_method(d_in=d_in, n_selections=n_selections)

    # Zero out the selector so masked features/mask don't affect
    # logits, then fix per-selection scores deterministically.
    with torch.no_grad():
        method.selector.weight.zero_()
        method.selector.bias.copy_(
            torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0])
        )

    masked_features = torch.zeros((2, d_in))
    feature_mask = torch.zeros((2, d_in), dtype=torch.bool)
    selection_mask = torch.zeros((2, n_selections), dtype=torch.bool)
    # Selections 0 (the grouped context selection) and 2 have
    # already been performed and must be excluded.
    selection_mask[:, [0, 2]] = True

    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=selection_mask,
        feature_shape=torch.Size((d_in,)),
    )

    # Highest-scoring non-excluded selection is index 1 (score 4.0),
    # returned as a 1-indexed selection.
    assert torch.equal(action, torch.tensor([[2], [2]]))


def test_gdfs_grouped_selection_requires_selection_mask() -> None:
    """A missing selection_mask must raise an explicit error."""
    d_in = 10
    n_contexts = 3
    n_selections = 1 + (d_in - n_contexts)  # 8
    method = _make_method(d_in=d_in, n_selections=n_selections)

    masked_features = torch.zeros((2, d_in))
    feature_mask = torch.zeros((2, d_in), dtype=torch.bool)

    with pytest.raises(ValueError, match="selection_mask"):
        method.act(
            masked_features=masked_features,
            feature_mask=feature_mask,
            selection_mask=None,
            feature_shape=torch.Size((d_in,)),
        )


def test_gdfs_grouped_selection_rejects_incompatible_selection_mask() -> None:
    d_in = 10
    n_contexts = 3
    n_selections = 1 + (d_in - n_contexts)  # 8
    method = _make_method(d_in=d_in, n_selections=n_selections)

    masked_features = torch.zeros((2, d_in))
    feature_mask = torch.zeros((2, d_in), dtype=torch.bool)
    wrong_shaped_selection_mask = torch.zeros((2, n_selections - 1))

    with pytest.raises(ValueError, match="selection_mask"):
        method.act(
            masked_features=masked_features,
            feature_mask=feature_mask,
            selection_mask=wrong_shaped_selection_mask.bool(),
            feature_shape=torch.Size((d_in,)),
        )


def test_gdfs_direct_unmasker_fallback_is_unchanged() -> None:
    """Direct Unmasker's feature-mask fallback keeps working."""
    d_in = 5
    n_selections = d_in
    method = _make_method(d_in=d_in, n_selections=n_selections)

    with torch.no_grad():
        method.selector.weight.zero_()
        method.selector.bias.copy_(torch.tensor([3.0, 5.0, 1.0, 4.0, 2.0]))

    masked_features = torch.zeros((2, d_in))
    feature_mask = torch.zeros((2, d_in), dtype=torch.bool)
    # Feature index 1 (the highest score) has already been observed.
    feature_mask[:, 1] = True

    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=None,
        feature_shape=torch.Size((d_in,)),
    )

    # Next highest-scoring unmasked feature is index 3 (score 4.0).
    assert torch.equal(action, torch.tensor([[4], [4]]))
