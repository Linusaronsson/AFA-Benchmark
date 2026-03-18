import math
from typing import ClassVar, cast, final, override

import numpy as np
import torch

from afabench.common.custom_types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
    SelectionMask,
)
from afabench.missing_values.masking import (
    MAR_mask,
    MCAR_mask,
    MNAR_mask_logistic,
    MNAR_mask_quantiles,
    MNAR_self_mask_logistic,
)


@final
class MissingnessInitializer(AFAInitializer):
    """
    Generate initial feature masks using MAR/MNAR missingness mechanisms.

    This initializer applies realistic missingness patterns from the malearn
    library (Stempfle et al.) to create warm-start masks for AFA evaluation.
    It enables evaluating AFA policies under realistic missing data scenarios.

    The missingness mask functions originate from:
    https://github.com/BorisMuzellec/MissingDataOT

    Note on mask convention: malearn returns True=missing, but AFA uses
    True=observed. This initializer inverts the mask accordingly.

    Supported mechanisms:
        - 'mcar': Missing Completely at Random
        - 'mar': Missing at Random via logistic masking model
        - 'mnar_logistic': Missing Not at Random via logistic model
        - 'mnar_self': MNAR with self-masking logistic model
        - 'mnar_quantiles': MNAR with quantile censorship
    """

    SUPPORTED_MECHANISMS: ClassVar[set[str]] = {
        "mcar",
        "mar",
        "mnar_logistic",
        "mnar_self",
        "mnar_quantiles",
    }
    _FORBIDDEN_MASK_SEED_OFFSET: ClassVar[int] = 9999
    _NONEMPTY_RESAMPLE_MAX_ATTEMPTS: ClassVar[int] = 1024

    def __init__(
        self,
        mechanism: str,
        p: float,
        p_obs: float = 0.3,
        p_params: float = 0.3,
        q: float = 0.25,
        cut: str = "both",
        *,
        exclude_inputs: bool = True,
        mcar: bool = False,
        acquirable_fraction: float = 1.0,
    ):
        """
        Initialize the missingness initializer.

        Args:
            mechanism: One of 'mcar', 'mar', 'mnar_logistic',
                'mnar_self', 'mnar_quantiles'.
            p: Proportion of missing values to generate.
            p_obs: (MAR only) Proportion of variables with no missing values.
            p_params: (MNAR logistic/quantiles) Proportion of variables used
                as inputs for the logistic model or with quantile censorship.
            q: (MNAR quantiles only) Quantile level for cuts.
            cut: (MNAR quantiles only) Where to apply cut: 'both', 'upper', 'lower'.
            exclude_inputs: (MNAR logistic only) Whether to exclude logistic
                model inputs from MNAR missingness.
            mcar: (MNAR quantiles only) Whether to add MCAR on top.
            acquirable_fraction: Fraction of missing features that are
                acquirable (the rest become never-acquirable). Must be
                in [0, 1]. Defaults to 1.0 (all missing features acquirable).
        """
        if mechanism not in self.SUPPORTED_MECHANISMS:
            msg = (
                f"Unknown mechanism: {mechanism}. "
                f"Supported: {self.SUPPORTED_MECHANISMS}"
            )
            raise ValueError(msg)
        if not (0.0 <= acquirable_fraction <= 1.0):
            msg = (
                "acquirable_fraction must be in [0, 1], got "
                f"{acquirable_fraction}."
            )
            raise ValueError(msg)

        self.mechanism = mechanism
        self.p = p
        self.p_obs = p_obs
        self.p_params = p_params
        self.q = q
        self.cut = cut
        self.exclude_inputs = exclude_inputs
        self.mcar = mcar
        self.acquirable_fraction = acquirable_fraction
        self.seed: int | None = None

    @override
    def set_seed(self, seed: int | None) -> None:
        self.seed = seed

    @override
    def initialize(
        self,
        features: Features,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> FeatureMask:
        assert feature_shape is not None, (
            "feature_shape must be provided for MissingnessInitializer"
        )

        # Flatten features to 2D (n, d) for the mask generation functions
        batch_shape = features.shape[: -len(feature_shape)]
        batch_size = math.prod(batch_shape)
        n_features = math.prod(feature_shape)

        flat_features = features.reshape(batch_size, n_features)

        # Generate missingness mask (True = missing in malearn convention),
        # then resample only the degenerate rows that would leave the sample
        # with no train-time support at all.
        missing_mask = self._generate_mask(flat_features, seed_offset=0)
        observed_mask = self._ensure_nonempty_support(
            features=flat_features,
            missing_mask=missing_mask,
        )

        # Reshape back to original batch + feature shape
        return observed_mask.reshape(batch_shape + feature_shape)

    def _effective_seed(self, seed_offset: int) -> int | None:
        if self.seed is None:
            return None
        return self.seed + seed_offset

    def _ensure_nonempty_support(
        self,
        *,
        features: torch.Tensor,
        missing_mask: torch.BoolTensor,
    ) -> torch.BoolTensor:
        """Resample fully-missing rows until every sample keeps some support."""
        observed_mask = cast("torch.BoolTensor", ~missing_mask)
        bad_rows = ~observed_mask.any(dim=1)
        if not bad_rows.any():
            return observed_mask

        for attempt in range(1, self._NONEMPTY_RESAMPLE_MAX_ATTEMPTS + 1):
            replacement_missing = self._generate_mask(
                features[bad_rows],
                seed_offset=attempt,
            )
            replacement_observed = cast(
                "torch.BoolTensor", ~replacement_missing
            )
            observed_mask[bad_rows] = replacement_observed
            bad_rows = ~observed_mask.any(dim=1)
            if not bad_rows.any():
                return observed_mask

        msg = (
            "Failed to sample non-empty train support for all rows after "
            f"{self._NONEMPTY_RESAMPLE_MAX_ATTEMPTS} attempts. "
            f"mechanism={self.mechanism!r}, p={self.p}."
        )
        raise RuntimeError(msg)

    def _generate_mask(
        self,
        x: torch.Tensor,
        *,
        seed_offset: int = 0,
    ) -> torch.BoolTensor:
        """
        Generate a missingness mask using the configured mechanism.

        Args:
            x: 2D tensor of shape (n, d).
            seed_offset: Offset added to the base seed for deterministic
                resampling attempts.

        Returns:
            Boolean tensor where True means missing (malearn convention).
        """
        # Use float64 for numerical stability in logistic fitting
        x_double = x.double()
        effective_seed = self._effective_seed(seed_offset)

        if self.mechanism == "mcar":
            mask = MCAR_mask(x_double, p=self.p, seed=effective_seed)
        elif self.mechanism == "mar":
            mask = MAR_mask(
                x_double,
                p=self.p,
                p_obs=self.p_obs,
                seed=effective_seed,
            )
        elif self.mechanism == "mnar_logistic":
            mask = MNAR_mask_logistic(
                x_double,
                p=self.p,
                p_params=self.p_params,
                exclude_inputs=self.exclude_inputs,
                seed=effective_seed,
            )
        elif self.mechanism == "mnar_self":
            mask = MNAR_self_mask_logistic(
                x_double,
                p=self.p,
                seed=effective_seed,
            )
        elif self.mechanism == "mnar_quantiles":
            mask = MNAR_mask_quantiles(
                x_double,
                p=self.p,
                q=self.q,
                p_params=self.p_params,
                cut=self.cut,
                MCAR=self.mcar,
                seed=effective_seed,
            )
        else:
            msg = f"Unknown mechanism: {self.mechanism}"
            raise ValueError(msg)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        return cast("torch.BoolTensor", mask.bool())

    @override
    def get_training_forbidden_mask(
        self,
        observed_mask: FeatureMask,
    ) -> FeatureMask:
        """
        Return mask where missing features are permanently forbidden.

        For training under incomplete data: features the initializer
        marks as missing (observed_mask=False) should never be visible
        or selectable during training.
        """
        return ~observed_mask

    def get_forbidden_selection_mask(
        self,
        observed_mask: FeatureMask,
        feature_shape: torch.Size,
    ) -> SelectionMask:
        """
        Compute a selection mask that marks never-acquirable features as forbidden.

        Among the missing features (observed_mask=False), randomly marks
        ``(1 - acquirable_fraction)`` of them as forbidden (True in selection mask).
        Already-observed features are never forbidden.

        This assumes a flat (1D) selection space matching the number of
        features, which is the common case for tabular data with per-feature
        unmaskers.

        Args:
            observed_mask: Boolean mask where True=observed. Shape: (*batch, *feature_shape).
            feature_shape: Shape of the features excluding the batch dimension.

        Returns:
            SelectionMask where True means the feature is forbidden (cannot be acquired).
            Shape: (*batch, n_features) where n_features = prod(feature_shape).
        """
        batch_shape = observed_mask.shape[: -len(feature_shape)]
        n_features = math.prod(feature_shape)
        flat_observed = observed_mask.reshape(*batch_shape, n_features)

        # Start with nothing forbidden
        forbidden = torch.zeros_like(flat_observed, dtype=torch.bool)

        if self.acquirable_fraction >= 1.0:
            return forbidden

        # For each sample, among missing features, randomly forbid some
        missing = ~flat_observed  # True where missing
        rng = torch.Generator(device=observed_mask.device)
        if self.seed is not None:
            rng.manual_seed(self.seed + self._FORBIDDEN_MASK_SEED_OFFSET)

        # Work on flattened batch
        flat_missing = missing.reshape(-1, n_features)
        flat_forbidden = forbidden.reshape(-1, n_features)
        batch_size = flat_missing.shape[0]

        for i in range(batch_size):
            missing_indices = torch.where(flat_missing[i])[0]
            n_missing = len(missing_indices)
            if n_missing == 0:
                continue
            n_to_block = round(n_missing * (1.0 - self.acquirable_fraction))
            if n_to_block == 0:
                continue
            perm = torch.randperm(
                n_missing, generator=rng, device=observed_mask.device
            )
            blocked_indices = missing_indices[perm[:n_to_block]]
            flat_forbidden[i, blocked_indices] = True

        return flat_forbidden.reshape(*batch_shape, n_features)
