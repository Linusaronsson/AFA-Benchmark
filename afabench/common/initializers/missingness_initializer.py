import torch
import numpy as np

from typing import final, override

from malearn.data.utils import (
    MAR_mask,
    MNAR_mask_logistic,
    MNAR_mask_quantiles,
    MNAR_self_mask_logistic,
)

from afabench.common.custom_types import (
    AFAInitializer,
    FeatureMask,
    Features,
    Label,
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
        - 'mar': Missing at Random via logistic masking model
        - 'mnar_logistic': Missing Not at Random via logistic model
        - 'mnar_self': MNAR with self-masking logistic model
        - 'mnar_quantiles': MNAR with quantile censorship
    """

    SUPPORTED_MECHANISMS = {
        "mar",
        "mnar_logistic",
        "mnar_self",
        "mnar_quantiles",
    }

    def __init__(
        self,
        mechanism: str,
        p: float,
        p_obs: float = 0.3,
        p_params: float = 0.3,
        q: float = 0.25,
        cut: str = "both",
        exclude_inputs: bool = True,
        mcar: bool = False,
    ):
        """
        Initialize the missingness initializer.

        Args:
            mechanism: One of 'mar', 'mnar_logistic', 'mnar_self', 'mnar_quantiles'.
            p: Proportion of missing values to generate.
            p_obs: (MAR only) Proportion of variables with no missing values.
            p_params: (MNAR logistic/quantiles) Proportion of variables used
                as inputs for the logistic model or with quantile censorship.
            q: (MNAR quantiles only) Quantile level for cuts.
            cut: (MNAR quantiles only) Where to apply cut: 'both', 'upper', 'lower'.
            exclude_inputs: (MNAR logistic only) Whether to exclude logistic
                model inputs from MNAR missingness.
            mcar: (MNAR quantiles only) Whether to add MCAR on top.
        """
        if mechanism not in self.SUPPORTED_MECHANISMS:
            msg = (
                f"Unknown mechanism: {mechanism}. "
                f"Supported: {self.SUPPORTED_MECHANISMS}"
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
        assert (
            feature_shape is not None
        ), "feature_shape must be provided for MissingnessInitializer"

        # Flatten features to 2D (n, d) for the mask generation functions
        batch_shape = features.shape[: -len(feature_shape)]
        batch_size = int(torch.prod(torch.tensor(batch_shape)).item())
        n_features = int(torch.prod(torch.tensor(feature_shape)).item())

        flat_features = features.reshape(batch_size, n_features)

        # Generate missingness mask (True = missing in malearn convention)
        missing_mask = self._generate_mask(flat_features)

        # Invert: malearn True=missing -> AFA True=observed
        observed_mask = ~missing_mask

        # Reshape back to original batch + feature shape
        return observed_mask.reshape(batch_shape + feature_shape)

    def _generate_mask(self, X: torch.Tensor) -> torch.BoolTensor:
        """
        Generate a missingness mask using the configured mechanism.

        Args:
            X: 2D tensor of shape (n, d).

        Returns:
            Boolean tensor where True means missing (malearn convention).
        """
        # Use float64 for numerical stability in logistic fitting
        X_double = X.double()

        if self.mechanism == "mar":
            mask = MAR_mask(
                X_double, p=self.p, p_obs=self.p_obs, seed=self.seed
            )
        elif self.mechanism == "mnar_logistic":
            mask = MNAR_mask_logistic(
                X_double,
                p=self.p,
                p_params=self.p_params,
                exclude_inputs=self.exclude_inputs,
                seed=self.seed,
            )
        elif self.mechanism == "mnar_self":
            mask = MNAR_self_mask_logistic(X_double, p=self.p, seed=self.seed)
        elif self.mechanism == "mnar_quantiles":
            mask = MNAR_mask_quantiles(
                X_double,
                p=self.p,
                q=self.q,
                p_params=self.p_params,
                cut=self.cut,
                MCAR=self.mcar,
                seed=self.seed,
            )
        else:
            msg = f"Unknown mechanism: {self.mechanism}"
            raise ValueError(msg)

        # Ensure output is a bool tensor
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)

        return mask.bool()
