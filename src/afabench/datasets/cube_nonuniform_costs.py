from typing import ClassVar, final, override

import torch

from afabench.core.types import Features, Label
from afabench.datasets.cube import CUBE_BINARY_CODES
from afabench.datasets.in_memory import InMemoryAFADataset
from afabench.datasets.utils import one_hot


@final
class CubeNonUniformCostsDataset(InMemoryAFADataset):
    """
    A modified version of CUBE that has two blocks of features which both contain informative features, but the blocks have different feature costs.

    This dataset can be used to verify whether an AFA method cares about feature costs.
    """

    config_keys: ClassVar[tuple[str, ...]] = (
        "n_samples",
        "seed",
        "non_informative_feature_mean",
        "non_informative_feature_std",
        "informative_feature_std",
        "cost_scaling",
    )
    # Number of cube features in each block
    n_cube_features: ClassVar[int] = 10
    n_blocks: ClassVar[int] = 2

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return True

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size([20])

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([8])

    def __init__(
        self,
        seed: int = 123,
        n_samples: int = 20000,
        non_informative_feature_mean: float = 0.5,
        informative_feature_std: float = 0.1,
        non_informative_feature_std: float = 0.3,
        cost_scaling: float = 2.0,
    ):
        self.seed = seed
        self.n_samples = n_samples
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.informative_feature_std = informative_feature_std
        self.cost_scaling = cost_scaling
        super().__init__()
        self.feature_costs = self._compute_feature_costs()

    @override
    def _build(self) -> tuple[Features, Label]:
        rng = torch.Generator()
        rng.manual_seed(self.seed)

        # Draw labels
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            dtype=torch.int64,
            generator=rng,
        )

        # Initialize feature blocks
        x_block1 = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=rng,
        )

        x_block2 = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=rng,
        )

        # Insert informative signals, in both blocks
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = CUBE_BINARY_CODES[lbl]

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            x_block1[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=rng,
                )
                + mu_bin
            )
            x_block2[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=rng,
                )
                + mu_bin
            )

        features = torch.cat([x_block1, x_block2], dim=1)
        return features, one_hot(y_int, self.label_shape[0])

    def _compute_feature_costs(self) -> torch.Tensor:
        block1_costs = torch.ones((self.n_cube_features,), dtype=torch.float32)
        block2_costs = self.cost_scaling * torch.ones((self.n_cube_features,))
        combined_costs = torch.cat([block1_costs, block2_costs], dim=-1)

        # Normalize so the average feature cost is still 1.0
        combined_costs = (
            self.n_features * combined_costs / combined_costs.sum()
        )
        return combined_costs

    @property
    def n_features(self) -> int:
        return self.n_cube_features * self.n_blocks
