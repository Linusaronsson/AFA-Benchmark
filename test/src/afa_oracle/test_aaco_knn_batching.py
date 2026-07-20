"""`get_knn_batched` against hand-computed shared-support distances."""

import pytest
import torch

from afabench.components.methods.oracle.aaco.core import get_knn_batched


def _reference(
    x_train: torch.Tensor,
    x_query: torch.Tensor,
    masks: torch.Tensor,
    observed: torch.Tensor | None,
) -> torch.Tensor:
    """
    Compute the distance AACO is specified to minimise, written out literally.

    Mean squared difference over the features the query has selected *and* the
    training row actually has, or +inf where that support is empty. Deliberately
    a slow triple loop over the direct `(x - q)^2` form: it is the specification
    the fast expanded/BLAS path has to match, not a second copy of it.
    """
    n_train, n_features = x_train.shape
    n_queries = x_query.shape[0]
    out = torch.full((n_train, n_queries), torch.inf, dtype=torch.float64)
    for i in range(n_train):
        for b in range(n_queries):
            total, count = 0.0, 0
            for j in range(n_features):
                if masks[j, b] <= 0 or (
                    observed is not None and not observed[i, j]
                ):
                    continue
                total += float(x_train[i, j] - x_query[b, j]) ** 2
                count += 1
            if count:
                out[i, b] = total / count
    return out


@pytest.mark.parametrize("restricted", [False, True])
def test_matches_the_specified_distance(*, restricted: bool) -> None:
    """Well-separated values, so exact ties cannot hide a genuine mismatch."""
    g = torch.Generator().manual_seed(0)
    n_train, n_features, n_queries, k = 40, 6, 5, 3
    # Distinct row scales keep every pairwise distance clearly separated.
    x_train = torch.arange(1, n_train + 1, dtype=torch.float32).unsqueeze(1)
    x_train = x_train * torch.randn(1, n_features, generator=g)
    x_query = torch.randn(n_queries, n_features, generator=g)
    masks = (torch.rand(n_features, n_queries, generator=g) > 0.5).float()
    masks[0] = 1.0
    observed = (
        (torch.rand(n_train, n_features, generator=g) > 0.3)
        if restricted
        else None
    )

    got = get_knn_batched(
        x_train, x_query, masks, k, train_observed_mask=observed
    )
    want = _reference(x_train, x_query, masks, observed).argsort(dim=0)[:k]

    assert got.shape == (k, n_queries)
    assert torch.equal(got, want)


def test_empty_shared_support_is_never_chosen() -> None:
    """
    A training row sharing no feature with the query must rank last.

    Row 0 is observed only where the query is not, so its distance is
    undefined; the oracle must not treat that as distance zero.
    """
    x_train = torch.tensor([[9.0, 9.0], [0.5, 9.0], [3.0, 9.0]])
    observed = torch.tensor([[False, True], [True, True], [True, True]])
    x_query = torch.tensor([[0.0, 0.0]])
    masks = torch.tensor([[1.0], [0.0]])  # query selects feature 0 only

    got = get_knn_batched(
        x_train, x_query, masks, 3, train_observed_mask=observed
    )

    assert got[0].item() == 1, "nearest on the shared support"
    assert got[1].item() == 2
    assert got[2].item() == 0, "empty shared support must rank last"


@pytest.mark.parametrize("restricted", [False, True])
def test_chunking_does_not_change_results(*, restricted: bool) -> None:
    """Chunk size is a memory knob and must not affect which neighbors win."""
    g = torch.Generator().manual_seed(1)
    n_train, n_features, n_queries, k = 200, 12, 16, 5
    x_train = torch.randn(n_train, n_features, generator=g)
    x_query = torch.randn(n_queries, n_features, generator=g)
    masks = (torch.rand(n_features, n_queries, generator=g) > 0.5).float()
    masks[0] = 1.0
    observed = (
        (torch.rand(n_train, n_features, generator=g) > 0.3)
        if restricted
        else None
    )

    unchunked = get_knn_batched(
        x_train,
        x_query,
        masks,
        k,
        batch_size=10_000,
        train_observed_mask=observed,
    )
    chunked = get_knn_batched(
        x_train, x_query, masks, k, batch_size=32, train_observed_mask=observed
    )
    assert torch.equal(unchunked, chunked)


def test_complete_data_ranking_is_unaffected_by_normalisation() -> None:
    """
    One formula serves complete and incomplete training data.

    For complete data the shared-support divisor is constant down each column,
    so it cannot reorder a column. This pins the algebra that lets the
    availability branch be the general case rather than a separate path.
    """
    g = torch.Generator().manual_seed(3)
    n_train, n_features, n_queries, k = 200, 12, 16, 5
    x_train = torch.arange(1, n_train + 1, dtype=torch.float32).unsqueeze(1)
    x_train = x_train * torch.randn(1, n_features, generator=g)
    x_query = torch.randn(n_queries, n_features, generator=g)
    masks = (torch.rand(n_features, n_queries, generator=g) > 0.5).float()
    masks[0] = 1.0

    normalised = get_knn_batched(x_train, x_query, masks, k)
    unnormalised = get_knn_batched(
        x_train,
        x_query,
        masks,
        k,
        train_observed_mask=torch.ones_like(x_train),
    )
    assert torch.equal(normalised, unnormalised)


def test_excluded_query_never_appears_in_its_own_neighbors() -> None:
    g = torch.Generator().manual_seed(2)
    n_train, n_features, n_queries, k = 200, 12, 16, 5
    x_train = torch.randn(n_train, n_features, generator=g)
    masks = (torch.rand(n_features, n_queries, generator=g) > 0.5).float()
    masks[0] = 1.0
    # Query b *is* train row b, so without exclusion it is its own neighbor.
    instance_idx = torch.arange(n_queries)
    x_query = x_train[:n_queries]

    kept = get_knn_batched(
        x_train, x_query, masks, k, instance_idx=instance_idx
    )
    dropped = get_knn_batched(
        x_train,
        x_query,
        masks,
        k,
        instance_idx=instance_idx,
        exclude_instance=True,
    )
    assert (kept == instance_idx.reshape(1, -1)).any()
    assert not (dropped == instance_idx.reshape(1, -1)).any()
