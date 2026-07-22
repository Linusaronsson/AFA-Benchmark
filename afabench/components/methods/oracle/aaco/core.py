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
from afabench.components.methods.rl.common.custom_types import (
    AFAFeatureRestorationFn,
)
from afabench.core.types import AFAClassifier
from afabench.core.utils import get_class_frequencies

logger = logging.getLogger(__name__)

MISSINGNESS_OBJECTIVES = {"support_aware", "doubly_robust"}


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
    Batching those makes cuBLAS select kernels by shape, so results acquire an
    `eval_batch_size` dependence. TF32 drift across batch sizes (2.6e-4) is
    within 1.4x of the smallest best-versus-runner-up gap (3.6e-4), so a
    decision could flip on batch size alone. Exactness costs 1.07x here.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def get_knn_batched(
    X_train: torch.Tensor,  # noqa: N803
    X_query: torch.Tensor,  # noqa: N803
    masks: torch.Tensor,
    num_neighbors: int,
    instance_idx: torch.Tensor | None = None,
    exclude_instance: bool = False,  # noqa: FBT002
    batch_size: int = 1000,
    train_observed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    K-NN over a batch of queries, one query mask per column.

    Follows the AACO paper's expanded distance form
    (https://github.com/lupalab/aaco/blob/3b2316661651699d11e904e9c5911c175e8b2fdc/src/aaco_rollout.py#L103),
    which takes a single `1 x d` query. Evaluating B instances that way issues
    B independent calls, each degenerating the matmuls to matrix-vector
    products split over `ceil(N/batch_size)`.

    Distance formula covers both complete and incomplete training data: the
    shared-support mean. Complete data is the special case where availability is
    all ones, and the mean then differs from the plain masked sum only by a
    factor constant down each column, which `topk` is invariant to.
    The normalisation is a necessary extension of the reference,
    because under missingness two training rows can share different numbers
    of features with the query and unnormalised sums are then not comparable.

    Args:
        X_train: N x d train instances
        X_query: B x d query instances
        masks: d x B binary masks, column b belonging to query b
        num_neighbors: number of neighbors (k)
        instance_idx: B-element indices of the query instances, for exclusion
        exclude_instance: whether to exclude each query from its own results
        batch_size: rows of X_train per chunk (memory bound)
        train_observed_mask: N x d availability mask, enabling the shared
            support distance used by the restricted strategies

    Returns:
        num_neighbors x B neighbor indices, column b belonging to query b.
    """
    n_rows = X_train.shape[0]
    masks = masks.to(X_train.device)
    # Fold each query's values into its own mask column so the per-query terms
    # become plain matrix products.
    weighted_query = X_query.T * masks  # (d, B), q_bj * m_jb
    weighted_query_squared = (X_query**2).T * masks  # (d, B), q_bj^2 * m_jb
    observed = (
        None
        if train_observed_mask is None
        else train_observed_mask.to(X_train.device).float()
    )

    dist_squared_chunks = []
    with _exact_matmul():
        for i in range(0, n_rows, batch_size):
            X_batch = X_train[i : i + batch_size]
            if observed is None:
                # Complete training data: every feature is available, so the
                # shared support is just the query mask and its size is
                # constant down the column.
                shared_counts = masks.sum(dim=0, keepdim=True)
                numerator = (
                    torch.matmul(X_batch**2, masks)
                    - 2.0 * torch.matmul(X_batch, weighted_query)
                    + weighted_query_squared.sum(dim=0, keepdim=True)
                )
            else:
                observed_batch = observed[i : i + batch_size]
                shared_counts = torch.matmul(observed_batch, masks)
                numerator = (
                    torch.matmul(X_batch**2 * observed_batch, masks)
                    - 2.0
                    * torch.matmul(X_batch * observed_batch, weighted_query)
                    + torch.matmul(observed_batch, weighted_query_squared)
                )
            dist_squared_chunks.append(
                (numerator / shared_counts.clamp_min(1)).masked_fill(
                    shared_counts == 0, torch.inf
                )
            )
    dist_squared = torch.cat(dist_squared_chunks, dim=0)  # (N, B)

    k = num_neighbors + int(exclude_instance)
    idx_topk = torch.topk(dist_squared, k, dim=0, largest=False)[1]  # (k, B)
    if not exclude_instance:
        return idx_topk
    assert instance_idx is not None
    # At most one entry per column is the query itself. A stable sort on the
    # "should drop" flag sinks it to the bottom while preserving topk order
    # among the rest, so slicing the top num_neighbors drops exactly it.
    drop = idx_topk == instance_idx.to(idx_topk.device).reshape(1, -1)
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
    future acquisition costs.
    """

    def __init__(
        self,
        k_neighbors: int = 5,
        acquisition_cost: float = 0.05,
        hide_val: float = 0.0,  # Use 0 for consistency with MLP training
        missingness_objective: str = "support_aware",
        dr_min_propensity: float = 1e-3,
        dr_max_weight: float | None = 20.0,
        mask_seed: int = 0,
        device: torch.device | None = None,
    ):
        if missingness_objective not in MISSINGNESS_OBJECTIVES:
            msg = (
                "missingness_objective must be one of "
                f"{sorted(MISSINGNESS_OBJECTIVES)}."
            )
            raise ValueError(msg)
        if not 0.0 < dr_min_propensity <= 1.0:
            msg = "dr_min_propensity must be in (0, 1]."
            raise ValueError(msg)
        if dr_max_weight is not None and dr_max_weight <= 0.0:
            msg = "dr_max_weight must be positive when provided."
            raise ValueError(msg)
        self.k_neighbors: int = k_neighbors
        self.acquisition_cost: float = acquisition_cost
        self.hide_val: float = hide_val
        self.missingness_objective: str = missingness_objective
        self.dr_min_propensity: float = dr_min_propensity
        self.dr_max_weight: float | None = dr_max_weight
        self.mask_seed: int = mask_seed
        self.classifier: AFAClassifier | None = None
        self.mask_generator: RandomMaskGenerator | None = None
        self._patch_mask_generators: dict[int, RandomMaskGenerator] = {}
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.train_observed_mask: torch.Tensor | None = None
        self.marginal_observation_probabilities: torch.Tensor | None = None
        self.device: torch.device = device or torch.device("cpu")
        self.class_weights: torch.Tensor | None = None
        self.feature_restoration_fn: AFAFeatureRestorationFn | None = None

    def fit(
        self,
        X_train: torch.Tensor,  # noqa: N803
        y_train: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
    ) -> None:
        """
        Fit the oracle on training data.

        Args:
            X_train: Training features (N x d)
            y_train: Training labels (N x n_classes), one-hot encoded
        """
        self.X_train = X_train.to(self.device)
        self.y_train = y_train.to(self.device)
        if observed_mask is None:
            self.train_observed_mask = None
            self.marginal_observation_probabilities = None
        else:
            observed_mask = observed_mask.to(self.device).bool()
            if observed_mask.shape != self.X_train.shape:
                msg = "observed_mask must have the same shape as X_train."
                raise ValueError(msg)
            self.train_observed_mask = (
                None if observed_mask.all() else observed_mask
            )
            self.marginal_observation_probabilities = (
                None
                if self.train_observed_mask is None
                else self.train_observed_mask.float().mean(dim=0)
            )

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

    def set_feature_restorer(
        self,
        feature_restoration_fn: AFAFeatureRestorationFn,
    ) -> None:
        """Set the label-free model used in online candidate backups."""
        self.feature_restoration_fn = feature_restoration_fn

    def to(self, device: torch.device) -> "AACOOracle":
        """Move oracle to device."""
        self.device = device
        if self.X_train is not None:
            self.X_train = self.X_train.to(device)
        if self.y_train is not None:
            self.y_train = self.y_train.to(device)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
        if self.train_observed_mask is not None:
            self.train_observed_mask = self.train_observed_mask.to(device)
        if self.marginal_observation_probabilities is not None:
            self.marginal_observation_probabilities = (
                self.marginal_observation_probabilities.to(device)
            )
        return self

    def _neighbor_observed_mask(
        self,
        neighbor_indices: torch.Tensor,
        feature_count: int,
    ) -> torch.Tensor:
        """Availability of each neighbor's features, shaped `(B, k, d)`."""
        if self.train_observed_mask is None:
            return torch.ones(
                (*neighbor_indices.shape, feature_count),
                dtype=torch.bool,
                device=self.device,
            )
        return self.train_observed_mask[neighbor_indices]

    def _neighbor_losses(
        self,
        neighbor_features: torch.Tensor,
        neighbor_labels: torch.Tensor,
        feature_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classifier loss per (instance, candidate mask, neighbor).

        Shapes are `(B, k, d)`, `(B, k, n_classes)` and `(B, M, k, d)` in, and
        `(B, M, k)` out. The instance dimension exists so that one classifier
        call covers a whole eval batch rather than one instance, which is where
        the launch-bound cost of this path used to sit.
        """
        assert self.classifier is not None
        n_masks = feature_masks.shape[1]
        candidate_features = neighbor_features.unsqueeze(1).expand(
            -1,
            n_masks,
            -1,
            -1,
        )
        return self._candidate_losses(
            candidate_features,
            neighbor_labels,
            feature_masks,
        )

    def _candidate_losses(
        self,
        candidate_features: torch.Tensor,
        neighbor_labels: torch.Tensor,
        feature_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Classifier loss for explicit `(instance, mask, neighbor)` values."""
        assert self.classifier is not None
        n_instances, n_masks, n_neighbors, feature_count = feature_masks.shape
        mask_float = feature_masks.float()
        masked = candidate_features * mask_float + self.hide_val * (
            1 - mask_float
        )
        with _exact_matmul():
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
        )
        if self.class_weights is not None:
            class_indices = neighbor_labels.argmax(dim=-1)
            losses = losses * self.class_weights[class_indices].unsqueeze(1)
        return losses

    def _stepwise_candidate_losses(
        self,
        candidate_feature_masks: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_labels: torch.Tensor,
        observed: torch.Tensor,
    ) -> torch.Tensor:
        """Restore unsupported candidate coordinates in one joint draw."""
        assert self.feature_restoration_fn is not None
        n_candidates = candidate_feature_masks.shape[1]
        candidate_masks = candidate_feature_masks.unsqueeze(2).expand(
            -1,
            -1,
            neighbor_features.shape[1],
            -1,
        )
        source_availability = observed.unsqueeze(1).expand_as(candidate_masks)
        candidate_values = neighbor_features.unsqueeze(1).expand(
            -1,
            n_candidates,
            -1,
            -1,
        )

        flat_values = candidate_values.flatten(end_dim=2)
        flat_candidates = candidate_masks.flatten(end_dim=2)
        flat_source = source_availability.flatten(end_dim=2)
        conditioning_mask = flat_candidates & flat_source
        restore_mask = flat_candidates & ~flat_source
        restore_rows = restore_mask.any(dim=1)
        restored_values = flat_values.clone()
        if restore_rows.any():
            inputs = flat_values[restore_rows].clone()
            inputs[~conditioning_mask[restore_rows]] = 0.0
            estimates = self.feature_restoration_fn(
                inputs,
                conditioning_mask[restore_rows],
            )
            row_values = restored_values[restore_rows]
            row_mask = restore_mask[restore_rows]
            row_values[row_mask] = estimates[row_mask]
            restored_values[restore_rows] = row_values

        restored_values = restored_values.view_as(candidate_values)
        return self._candidate_losses(
            restored_values,
            neighbor_labels,
            candidate_masks,
        )

    def _candidate_support_propensities(
        self,
        candidate_feature_masks: torch.Tensor,
    ) -> torch.Tensor:
        if self.marginal_observation_probabilities is None:
            return torch.ones(
                candidate_feature_masks.shape[:-1],
                device=self.device,
            )
        marginal = self.marginal_observation_probabilities.clamp_min(1e-12)
        log_propensity = (
            candidate_feature_masks.float() * marginal.log()
        ).sum(dim=-1)
        return log_propensity.exp()

    def _expected_candidate_losses(
        self,
        candidate_feature_masks: torch.Tensor,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score every candidate mask against its instance's neighbors.

        `candidate_feature_masks` is `(B, M, d)` and `neighbor_indices` is
        `(B, k)`; the result is `(B, M)`. Callers with a single instance pass
        `B == 1`.
        """
        assert self.X_train is not None
        assert self.y_train is not None
        candidate_feature_masks = candidate_feature_masks.bool()
        neighbor_features = self.X_train[neighbor_indices]
        neighbor_labels = self.y_train[neighbor_indices]
        observed = self._neighbor_observed_mask(
            neighbor_indices,
            candidate_feature_masks.shape[-1],
        )
        if self.feature_restoration_fn is not None:
            return self._stepwise_candidate_losses(
                candidate_feature_masks,
                neighbor_features,
                neighbor_labels,
                observed,
            ).mean(dim=-1)
        overlap_masks = candidate_feature_masks.unsqueeze(
            2
        ) & observed.unsqueeze(1)
        overlap_losses = self._neighbor_losses(
            neighbor_features,
            neighbor_labels,
            overlap_masks,
        )
        if (
            self.missingness_objective != "doubly_robust"
            or self.train_observed_mask is None
        ):
            return overlap_losses.mean(dim=-1)

        full_masks = candidate_feature_masks.unsqueeze(2).expand_as(
            overlap_masks
        )
        full_losses = self._neighbor_losses(
            neighbor_features,
            neighbor_labels,
            full_masks,
        )
        fully_supported = (
            (~candidate_feature_masks.unsqueeze(2)) | observed.unsqueeze(1)
        ).all(dim=-1)
        inverse_weights = (
            self._candidate_support_propensities(candidate_feature_masks)
            .clamp_min(self.dr_min_propensity)
            .reciprocal()
        )
        if self.dr_max_weight is not None:
            inverse_weights = inverse_weights.clamp_max(self.dr_max_weight)
        baseline = overlap_losses.mean(dim=-1, keepdim=True)
        corrected = baseline + fully_supported.float() * (
            inverse_weights.unsqueeze(-1) * (full_losses - baseline)
        )
        return corrected.mean(dim=-1)

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
        instance_idx: torch.Tensor | None = None,
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

        Candidate masks are not deduplicated. The generator
        ignores the current mask, so `maximum(new_masks, current)` is
        rectangular across instances and only stays that way without a
        per-instance `unique`. Duplicates cost a few redundant classifier rows,
        which are free once the call is batched, and they cannot change which
        mask wins, only which of several identical copies of it does.
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
            instance_idx=(
                torch.arange(batch_size, device=device)
                if instance_idx is None
                else instance_idx.to(device)
            ),
            exclude_instance=exclude_instance,
            train_observed_mask=self.train_observed_mask,
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
        )
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
            # Stopping is not allowed, so
            # fall back to the first unacquired selection.
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
        instance_idx: torch.Tensor | None = None,
        force_acquisition: bool = False,
        exclude_instance: bool = True,
    ) -> list[int | None]:
        """
        Select the next **selection** (not feature) for each instance in a batch.

        This is the path for unmaskers whose selections are not individual
        features, such as `CubeNMUnmasker` grouping the context features. It is
        a greedy one-step search over selections rather than the Monte Carlo
        candidate-mask search the feature-level oracle runs, so the two are
        genuinely different algorithms and not two spellings of one.

        `x_observed` and `observed_mask` are `(B, d)`, `selection_mask` is
        `(B, S)` and `selection_to_feature_mask` is `(S, d)`. Returns one
        selection index per instance, or None where the oracle prefers to stop.

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
            instance_idx=(
                torch.arange(batch_size, device=device)
                if instance_idx is None
                else instance_idx.to(device)
            ),
            exclude_instance=exclude_instance,
            train_observed_mask=self.train_observed_mask,
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
        # An already-taken selection is not a candidate, and stopping is not one
        # either under a hard budget. Both drop out as infinities rather than as
        # branches, which is what keeps the pass rectangular.
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
