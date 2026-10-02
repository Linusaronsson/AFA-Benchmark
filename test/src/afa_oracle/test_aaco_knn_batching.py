"""`get_knn_batched` against hand-computed masked squared distances."""

import torch

from afabench.components.methods.oracle.aaco.core import get_knn_batched


def _reference(
    x_train: torch.Tensor,
    x_query: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the distance AACO is specified to minimise, written out literally.

    Squared difference summed over the features the query has observed.
    Deliberately a slow triple loop over the direct `(x - q)^2` form: it is
    the specification the fast expanded/BLAS path has to match, not a second
    copy of it.
    """
    n_train, n_features = x_train.shape
    n_queries = x_query.shape[0]
    out = torch.zeros((n_train, n_queries), dtype=torch.float64)
    for i in range(n_train):
        for b in range(n_queries):
            total = 0.0
            for j in range(n_features):
                if masks[j, b] <= 0:
                    continue
                total += float(x_train[i, j] - x_query[b, j]) ** 2
            out[i, b] = total
    return out


def test_matches_the_specified_distance() -> None:
    """Well-separated values, so exact ties cannot hide a genuine mismatch."""
    g = torch.Generator().manual_seed(0)
    n_train, n_features, n_queries, k = 40, 6, 5, 3
    # Distinct row scales keep every pairwise distance clearly separated.
    x_train = torch.arange(1, n_train + 1, dtype=torch.float32).unsqueeze(1)
    x_train = x_train * torch.randn(1, n_features, generator=g)
    x_query = torch.randn(n_queries, n_features, generator=g)
    masks = (torch.rand(n_features, n_queries, generator=g) > 0.5).float()
    masks[0] = 1.0

    got = get_knn_batched(x_train, x_query, masks, k)
    want = _reference(x_train, x_query, masks).argsort(dim=0)[:k]

    assert got.shape == (k, n_queries)
    assert torch.equal(got, want)


def test_chunking_does_not_change_results() -> None:
    """Chunk size is a memory knob and must not affect which neighbors win."""
    g = torch.Generator().manual_seed(1)
    n_train, n_features, n_queries, k = 200, 12, 16, 5
    x_train = torch.randn(n_train, n_features, generator=g)
    x_query = torch.randn(n_queries, n_features, generator=g)
    masks = (torch.rand(n_features, n_queries, generator=g) > 0.5).float()
    masks[0] = 1.0

    unchunked = get_knn_batched(x_train, x_query, masks, k, batch_size=10_000)
    chunked = get_knn_batched(x_train, x_query, masks, k, batch_size=32)
    assert torch.equal(unchunked, chunked)


def test_batching_queries_matches_one_query_at_a_time() -> None:
    """Each column of the batched result is that query's own neighbours."""
    g = torch.Generator().manual_seed(4)
    n_train, n_features, n_queries, k = 120, 10, 9, 4
    x_train = torch.arange(1, n_train + 1, dtype=torch.float32).unsqueeze(1)
    x_train = x_train * torch.randn(1, n_features, generator=g)
    x_query = torch.randn(n_queries, n_features, generator=g)
    masks = (torch.rand(n_features, n_queries, generator=g) > 0.5).float()
    masks[0] = 1.0

    batched = get_knn_batched(x_train, x_query, masks, k)
    single = torch.cat(
        [
            get_knn_batched(
                x_train, x_query[b : b + 1], masks[:, b : b + 1], k
            )
            for b in range(n_queries)
        ],
        dim=1,
    )
    assert torch.equal(batched, single)


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
    assert dropped.shape == (k, n_queries)
