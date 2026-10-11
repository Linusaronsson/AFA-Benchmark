import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import NamedTuple

import torch
import torch.nn.functional as F

from afabench.components.methods.oracle.aaco.mask_generator import (
    RandomMaskGenerator,
    random_mask_generator,
)
from afabench.components.methods.oracle.aaco.utils import (
    ensure_probabilities,
    get_patch_dimensions,
    uses_patch_selection,
)
from afabench.core.types import AFAClassifier
from afabench.core.utils import get_class_frequencies

logger = logging.getLogger(__name__)


class _SelectionSpace(NamedTuple):
    """
    What the oracle is choosing over, and how that maps onto features.

    The feature-level oracle picks single features, so the two spaces coincide
    and both projections are the identity. The patch-level oracle picks square
    patches, so a selection covers many features at once.
    """

    selection_dim: int
    mask_generator: RandomMaskGenerator
    to_feature_mask: Callable[[torch.Tensor], torch.Tensor]
    to_selection_mask: Callable[[torch.Tensor], torch.Tensor]


@contextmanager
def _exact_matmul() -> Generator[None]:
    """
    Disable TF32 for matmuls whose output feeds a discrete choice.

    Wraps the KNN distances and the candidate-scoring classifier forward.
    Batching those makes cuBLAS select kernels by shape, so results would
    otherwise acquire an evaluation batch size dependence: TF32 drift across
    batch sizes is of the same order as the smallest best-versus-runner-up
    gap, so a decision could flip on batch size alone.
    """
    # Round-trip the precision setting rather than the raw TF32 flag: the
    # flag cannot represent "medium", so restoring only it would leave a
    # process-wide "medium" precision downgraded to "high" afterwards.
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous_precision)


def get_knn_batched(
    X_train: torch.Tensor,  # noqa: N803
    X_query: torch.Tensor,  # noqa: N803
    masks: torch.Tensor,
    num_neighbors: int,
    split_index: torch.Tensor | None = None,
    exclude_instance: bool = False,  # noqa: FBT002
    batch_size: int = 1000,
) -> torch.Tensor:
    """
    K-NN over a batch of queries, one query mask per column.

    Follows the AACO paper's expanded distance form
    (https://github.com/lupalab/aaco/blob/3b2316661651699d11e904e9c5911c175e8b2fdc/src/aaco_rollout.py#L103),
    which takes a single `1 x d` query. Evaluating B instances that way issues
    B independent calls, each degenerating the matmuls to matrix-vector
    products split over `ceil(N/batch_size)`. Folding each query's values
    into its own mask column turns the per-query terms into plain matrix
    products, so the whole batch is one pass over the training data.

    Args:
        X_train: N x d train instances
        X_query: B x d query instances
        masks: d x B binary masks, column b belonging to query b
        num_neighbors: number of neighbors (k)
        split_index: B-element indices of the query instances, for exclusion
        exclude_instance: whether to exclude each query from its own results
        batch_size: rows of X_train per chunk (memory bound)

    Returns:
        num_neighbors x B neighbor indices, column b belonging to query b.
    """
    n_rows = X_train.shape[0]
    masks = masks.to(X_train.device)
    weighted_query = X_query.T * masks  # (d, B), q_bj * m_jb
    query_term = ((X_query**2).T * masks).sum(dim=0, keepdim=True)  # (1, B)

    dist_squared_chunks = []
    with _exact_matmul():
        for i in range(0, n_rows, batch_size):
            X_batch = X_train[i : i + batch_size]
            dist_squared_chunks.append(
                torch.matmul(X_batch**2, masks)
                - 2.0 * torch.matmul(X_batch, weighted_query)
                + query_term
            )
    dist_squared = torch.cat(dist_squared_chunks, dim=0)  # (N, B)

    k = num_neighbors + int(exclude_instance)
    idx_topk = torch.topk(dist_squared, k, dim=0, largest=False)[1]  # (k, B)
    if not exclude_instance:
        return idx_topk
    assert split_index is not None
    # At most one entry per column is the query itself. A stable sort on the
    # "should drop" flag sinks it to the bottom while preserving topk order
    # among the rest, so slicing the top num_neighbors drops exactly it.
    drop = idx_topk == split_index.to(idx_topk.device).reshape(1, -1)
    order = torch.argsort(drop.int(), dim=0, stable=True)
    return idx_topk.gather(0, order)[:num_neighbors]


def load_mask_generator(input_dim: int, seed: int) -> RandomMaskGenerator:
    """Their exact mask generator loading logic."""
    # Paper shows this works nearly as well as 10,000 (for MNIST)
    return random_mask_generator(100, input_dim, 100, seed)


class AACOOracle:
    """
    Acquisition Conditioned Oracle for non-greedy active feature acquisition.

    This oracle implements the AACO algorithm from Valancius et al. 2024.
    (https://proceedings.mlr.press/v235/valancius24a.html)

    It selects features by optimizing a non-greedy objective that considers
    future acquisition costs. Every selection method scores a whole batch of
    instances with one neighbour search and one classifier call.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        acquisition_cost: float = 0.05,
        hide_val: float = 0.0,  # Use 0 for consistency with MLP training
        mask_seed: int = 0,
        device: torch.device | None = None,
    ):
        self.k_neighbors: int = k_neighbors
        self.acquisition_cost: float = acquisition_cost
        self.hide_val: float = hide_val
        self.mask_seed: int = mask_seed
        self.classifier: AFAClassifier | None = None
        self.mask_generator: RandomMaskGenerator | None = None
        self._patch_mask_generators: dict[int, RandomMaskGenerator] = {}
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.device: torch.device = device or torch.device("cpu")
        self.class_weights: torch.Tensor | None = None

    def fit(self, X_train: torch.Tensor, y_train: torch.Tensor) -> None:  # noqa: N803
        """
        Fit the oracle on training data.

        Args:
            X_train: Training features (N x d)
            y_train: Training labels (N x n_classes), one-hot encoded
        """
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)

        train_class_probabilities = get_class_frequencies(self.y_train)
        self.class_weights = len(train_class_probabilities) / (
            len(train_class_probabilities) * train_class_probabilities
        )

        input_dim = X_train.shape[1]
        self.mask_generator = load_mask_generator(input_dim, self.mask_seed)

        logger.info(f"Training data: {X_train.shape}")

    def set_classifier(self, classifier: AFAClassifier) -> None:
        """Set the classifier model used by the oracle."""
        self.classifier = classifier

    def to(self, device: torch.device) -> "AACOOracle":
        """Move oracle to device."""
        self.device = device
        if self.X_train is not None:
            self.X_train = self.X_train.to(device)
        if self.y_train is not None:
            self.y_train = self.y_train.to(device)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
        return self

    def _expected_candidate_losses(
        self,
        candidate_feature_masks: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score every candidate mask against its instance's neighbors.

        `candidate_feature_masks` is `(B, M, d)` and `neighbor_indices` is
        `(B, k)`; the result is `(B, M)`, the class-weighted cross-entropy of
        the classifier on each neighbor restricted to the candidate mask,
        averaged over neighbors. The instance dimension exists so that one
        classifier call covers a whole evaluation batch rather than one
        instance, which is where the launch-bound cost of this path used to
        sit. Callers with a single instance pass `B == 1`.
        """
        assert self.classifier is not None
        assert self.X_train is not None
        assert self.y_train is not None
        n_instances, n_masks, feature_count = candidate_feature_masks.shape
        n_neighbors = neighbor_indices.shape[1]
        neighbor_features = self.X_train[neighbor_indices]  # (B, k, d)
        neighbor_labels = self.y_train[neighbor_indices]  # (B, k, C)

        mask_float = (
            candidate_feature_masks.float()
            .unsqueeze(2)
            .expand(-1, -1, n_neighbors, -1)
        )  # (B, M, k, d)
        masked = neighbor_features.unsqueeze(1)
        masked = masked * mask_float + self.hide_val * (1 - mask_float)
        with torch.no_grad(), _exact_matmul():
            logits = self.classifier(
                masked.reshape(-1, feature_count),
                mask_float.reshape(-1, feature_count),
                feature_shape=torch.Size([feature_count]),
            )
        probabilities = ensure_probabilities(logits).view(
            n_instances,
            n_masks,
            n_neighbors,
            -1,
        )
        losses = -torch.sum(
            neighbor_labels.unsqueeze(1) * torch.log(probabilities + 1e-10),
            dim=-1,
        )  # (B, M, k)
        if self.class_weights is not None:
            class_indices = neighbor_labels.argmax(dim=-1)
            losses = losses * self.class_weights[class_indices].unsqueeze(1)
        return losses.mean(dim=-1)

    def _selection_space(
        self,
        feature_count: int,
        feature_shape: torch.Size | None,
        selection_size: int | None,
    ) -> _SelectionSpace:
        """Resolve the space the oracle selects over. See `_SelectionSpace`."""
        if not uses_patch_selection(selection_size, feature_shape):
            assert self.mask_generator is not None
            return _SelectionSpace(
                selection_dim=feature_count,
                mask_generator=self.mask_generator,
                to_feature_mask=lambda masks: masks.bool(),
                to_selection_mask=lambda masks: masks,
            )

        assert feature_shape
        assert selection_size is not None
        n_channels, height, width, patch_h, patch_w = get_patch_dimensions(
            selection_size, feature_shape
        )
        mask_width = int(selection_size**0.5)

        generator = self._patch_mask_generators.get(selection_size)
        if generator is None:
            generator = random_mask_generator(
                100, selection_size, 100, self.mask_seed
            )
            self._patch_mask_generators[selection_size] = generator

        def to_feature_mask(masks: torch.Tensor) -> torch.Tensor:
            leading = masks.shape[:-1]
            patches = masks.reshape(-1, 1, mask_width, mask_width).float()
            patches = F.interpolate(
                patches,
                scale_factor=(patch_h, patch_w),
                mode="nearest-exact",
            )
            if n_channels > 1:
                patches = patches.expand(-1, n_channels, height, width)
            return patches.reshape(*leading, feature_count).bool()

        def to_selection_mask(masks: torch.Tensor) -> torch.Tensor:
            # A patch counts as selected once any of its features is observed.
            grid = masks.view(
                -1, n_channels, mask_width, patch_h, mask_width, patch_w
            )
            return grid.any(dim=(1, 3, 5)).reshape(masks.shape[0], -1)

        return _SelectionSpace(
            selection_dim=selection_size,
            mask_generator=generator,
            to_feature_mask=to_feature_mask,
            to_selection_mask=to_selection_mask,
        )

    def select_next_features_batched(
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
        *,
        split_index: torch.Tensor | None = None,
        force_acquisition: bool = False,
        exclude_instance: bool = True,
        feature_shape: torch.Size | None = None,
        selection_size: int | None = None,
        selection_costs: torch.Tensor | None = None,
        selection_mask: torch.Tensor | None = None,
    ) -> list[int | None]:
        """
        Select the next feature to acquire for each instance in a batch.

        `x_observed` and `observed_mask` are `(B, d)`. Returns one selection
        index per instance, or None where the oracle prefers to stop.
        The whole batch shares one KNN and one classifier call.

        Candidate masks are not deduplicated. The generator ignores the
        current mask, so `maximum(new_masks, current)` is rectangular across
        instances and only stays that way without a per-instance `unique`.
        Duplicates cost a few redundant classifier rows, which are cheap once
        the call is batched. Dropping the `unique` (which also sorted the
        candidates) means an exact loss tie between distinct masks is now
        broken in generation order instead of sorted order; the winning loss
        is unchanged.

        Args:
            x_observed: masked features, unobserved entries equal to hide_val
            observed_mask: feature mask per instance
            split_index: training-set index of each instance, for exclusion
            force_acquisition: if True, never stop while a selection is left
            exclude_instance: whether to drop each instance from its own
                neighbours
            feature_shape: feature shape, needed for patch selection
            selection_size: number of selections for patch selection
            selection_costs: optional per-selection costs, overriding the
                unit-cost penalty
            selection_mask: optional `(B, S)` selection mask; when absent it
                is derived from `observed_mask`
        """
        assert self.classifier is not None, (
            "Oracle must have a classifier set. Call set_classifier() first."
        )
        assert self.X_train is not None, (
            "Oracle must be fitted first. Call fit() first."
        )
        assert self.y_train is not None, (
            "Oracle must be fitted first. Call fit() first."
        )

        device = self.device
        x_observed = x_observed.to(device)
        observed_feature_mask = observed_mask.to(device).bool()
        batch_size, feature_count = observed_feature_mask.shape

        idx_nn = get_knn_batched(
            self.X_train,
            x_observed,
            observed_feature_mask.float().T,
            self.k_neighbors,
            split_index=(
                torch.arange(batch_size, device=device)
                if split_index is None
                else split_index.to(device)
            ),
            exclude_instance=exclude_instance,
        ).T

        space = self._selection_space(
            feature_count, feature_shape, selection_size
        )

        if selection_mask is not None:
            current_selection_mask = (
                selection_mask.to(device).bool().reshape(batch_size, -1)
            )
            assert current_selection_mask.shape[1] == space.selection_dim, (
                "selection_mask has incompatible selection dimension."
            )
        else:
            current_selection_mask = space.to_selection_mask(
                observed_feature_mask
            )

        current_selection_float = current_selection_mask.float()

        new_masks = space.mask_generator(current_selection_float).to(device)
        candidate_selection_masks = torch.maximum(
            new_masks.unsqueeze(0), current_selection_float.unsqueeze(1)
        )  # (B, M, S)
        if not force_acquisition:
            # Slot 0 is the option to acquire nothing further, i.e. to stop.
            candidate_selection_masks[:, 0] = current_selection_float

        candidate_feature_masks = space.to_feature_mask(
            candidate_selection_masks
        ) | observed_feature_mask.unsqueeze(1)
        expected_losses = self._expected_candidate_losses(
            candidate_feature_masks,
            idx_nn,
        )

        # Add acquisition cost penalty.
        if selection_costs is not None:
            newly_selected = candidate_selection_masks.bool() & ~(
                current_selection_mask.unsqueeze(1)
            )
            acquisition_penalty = (
                newly_selected.float() * selection_costs.to(device)
            ).sum(dim=-1)
        else:
            acquisition_penalty = candidate_selection_masks.sum(
                dim=-1
            ) - current_selection_float.sum(dim=-1, keepdim=True)

        costs = expected_losses + self.acquisition_cost * acquisition_penalty
        best_idx = costs.argmin(dim=1)
        best_selection_mask = candidate_selection_masks[
            torch.arange(batch_size, device=device), best_idx
        ].bool()

        new_selections = best_selection_mask & ~current_selection_mask
        n_new = new_selections.sum(dim=1)

        chosen = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        chosen = torch.where(
            n_new == 1, new_selections.int().argmax(dim=1), chosen
        )

        needs_tiebreak = n_new > 1
        if bool(needs_tiebreak.any()):
            # Tie-break: of the selections in the winning subset, take the one
            # that most reduces expected loss when added alone.
            eye = torch.eye(
                space.selection_dim, dtype=torch.bool, device=device
            )
            ordering_feature_masks = space.to_feature_mask(
                current_selection_mask.unsqueeze(1) | eye
            ) | observed_feature_mask.unsqueeze(1)
            ordering_losses = self._expected_candidate_losses(
                ordering_feature_masks,
                idx_nn,
            ).masked_fill(~new_selections, torch.inf)
            chosen = torch.where(
                needs_tiebreak, ordering_losses.argmin(dim=1), chosen
            )

        if force_acquisition:
            # Stopping is not allowed, so fall back to the first unacquired
            # selection.
            unselected = ~current_selection_mask
            chosen = torch.where(
                (n_new == 0) & unselected.any(dim=1),
                unselected.int().argmax(dim=1),
                chosen,
            )

        return [None if c < 0 else c for c in chosen.tolist()]

    def select_next_selections_batched(
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
        selection_mask: torch.Tensor,
        selection_to_feature_mask: torch.Tensor,
        selection_costs: torch.Tensor | None = None,
        *,
        split_index: torch.Tensor | None = None,
        force_acquisition: bool = False,
        exclude_instance: bool = True,
    ) -> list[int | None]:
        """
        Select the next **selection** (not feature) for each instance in a batch.

        This is the path for unmaskers whose selections are not individual
        features, such as `CubeNMUnmasker` grouping the context features. It
        is a greedy one-step search over selections rather than the Monte
        Carlo candidate-mask search the feature-level oracle runs, so the two
        are genuinely different algorithms and not two spellings of one.

        `x_observed` and `observed_mask` are `(B, d)`, `selection_mask` is
        `(B, S)` and `selection_to_feature_mask` is `(S, d)`. Returns one
        selection index per instance, or None where the oracle prefers to
        stop.

        Every instance scores all S selections, including the ones it has
        already taken, whose cost is then set to infinity. That wastes a few
        classifier rows but keeps the candidate set rectangular across the
        batch, which is what lets the whole batch share one KNN and one
        classifier call. Slot 0 is the option to stop, so it stays ahead of
        every selection and ties resolve on the lowest selection index.
        """
        assert self.classifier is not None, (
            "Oracle must have a classifier set. Call set_classifier() first."
        )
        assert self.X_train is not None, (
            "Oracle must be fitted first. Call fit() first."
        )
        assert self.y_train is not None, (
            "Oracle must be fitted first. Call fit() first."
        )
        assert selection_to_feature_mask.ndim == 2, (
            "selection_to_feature_mask must be 2D: (n_selections, n_features)"
        )

        device = self.device
        x_observed = x_observed.to(device)
        observed_feature_mask = observed_mask.to(device).bool()
        current_selection_mask = selection_mask.to(device).bool()
        selection_to_feature_mask = selection_to_feature_mask.to(device).bool()
        batch_size, feature_count = observed_feature_mask.shape

        assert selection_to_feature_mask.shape[1] == feature_count, (
            "selection_to_feature_mask has incompatible feature dimension."
        )

        idx_nn = get_knn_batched(
            self.X_train,
            x_observed,
            observed_feature_mask.float().T,
            self.k_neighbors,
            split_index=(
                torch.arange(batch_size, device=device)
                if split_index is None
                else split_index.to(device)
            ),
            exclude_instance=exclude_instance,
        ).T

        # Slot 0 is "acquire nothing further", slot 1 + s is "acquire s".
        base_feature_mask = observed_feature_mask.unsqueeze(1)  # (B, 1, d)
        candidate_feature_masks = torch.cat(
            [
                base_feature_mask,
                base_feature_mask | selection_to_feature_mask.unsqueeze(0),
            ],
            dim=1,
        )
        expected_losses = self._expected_candidate_losses(
            candidate_feature_masks,
            idx_nn,
        )

        if selection_costs is not None:
            candidate_costs = (
                selection_costs.to(device)
                .float()
                .reshape(1, -1)
                .expand(batch_size, -1)
            )
        else:
            candidate_costs = (
                (selection_to_feature_mask.unsqueeze(0) & ~base_feature_mask)
                .sum(dim=-1)
                .float()
            )
        candidate_costs = torch.cat(
            [torch.zeros((batch_size, 1), device=device), candidate_costs],
            dim=1,
        )

        costs = expected_losses + self.acquisition_cost * candidate_costs
        # An already-taken selection is not a candidate, and stopping is not
        # one either under a hard budget. Both drop out as infinities rather
        # than as branches, which is what keeps the pass rectangular.
        stop_unavailable = torch.full(
            (batch_size, 1),
            force_acquisition,
            dtype=torch.bool,
            device=device,
        )
        costs = costs.masked_fill(
            torch.cat([stop_unavailable, current_selection_mask], dim=1),
            torch.inf,
        )

        # Slot 0 wins where nothing is left, which is the honest "stop" answer
        # even under force_acquisition, and matches -1 below.
        chosen = costs.argmin(dim=1) - 1
        return [None if c < 0 else c for c in chosen.tolist()]

    def predict_with_mask(
        self,
        x_observed: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Make prediction given observed features.

        Args:
            x_observed: 1D tensor of observed features
            observed_mask: 1D boolean tensor indicating which features are observed

        Returns:
            Class probabilities (n_classes,)
        """
        if self.classifier is None:
            msg = "Oracle must have a classifier set."
            raise ValueError(msg)

        x_masked = x_observed.unsqueeze(0).to(self.device)
        mask = observed_mask.float().unsqueeze(0).to(self.device)

        # Apply masking
        x_input = x_masked * mask + self.hide_val * (1 - mask)

        with torch.no_grad():
            feature_shape = torch.Size([x_input.shape[1]])
            logits = self.classifier(
                x_input, mask, feature_shape=feature_shape
            )
            probs = ensure_probabilities(logits)

        return probs.squeeze(0)
