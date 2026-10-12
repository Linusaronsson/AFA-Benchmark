from typing import ClassVar, final, override

import torch
from jaxtyping import Integer
from torch import Tensor

from afabench.core.types import Features, Label
from afabench.datasets.in_memory import InMemoryAFADataset
from afabench.datasets.utils import one_hot

type CubeBinaryCodes = Integer[Tensor, "8 3"]

# The 3-bit binary code of each of the 8 CUBE labels, least significant
# bit first
CUBE_BINARY_CODES: CubeBinaryCodes = torch.stack(
    [torch.tensor([int(b) for b in format(i, "03b")]) for i in range(8)],
    dim=0,
).flip(-1)


@final
class CubeDataset(InMemoryAFADataset):
    """
    The Cube dataset, as described in the paper "ODIN: Optimal Discovery of High-value INformation Using Model-based Deep Reinforcement Learning".

    Implements the AFADataset protocol.
    """

    config_keys: ClassVar[tuple[str, ...]] = (
        "n_samples",
        "seed",
        "non_informative_feature_mean",
        "non_informative_feature_std",
        "informative_feature_std",
    )
    n_cube_features: ClassVar[int] = 10
    # Remaining features are dummy features
    n_dummy_features: ClassVar[int] = 10

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
    ):
        self.seed = seed
        self.n_samples = n_samples
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.informative_feature_std = informative_feature_std
        super().__init__()

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
        x_cube = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_cube_features),
            generator=rng,
        )

        x_dummy = torch.normal(
            mean=self.non_informative_feature_mean,
            std=self.non_informative_feature_std,
            size=(self.n_samples, self.n_dummy_features),
            generator=rng,
        )

        # Insert informative signals
        for i in range(self.n_samples):
            lbl = int(y_int[i].item())
            mu_bin = CUBE_BINARY_CODES[lbl]

            # Cube features: 3 bumps
            idxs = [(lbl + j) for j in range(3)]
            x_cube[i, idxs] = (
                torch.normal(
                    mean=0.0,
                    std=self.informative_feature_std,
                    size=(3,),
                    generator=rng,
                )
                + mu_bin
            )

        features = torch.cat([x_cube, x_dummy], dim=1)
        return features, one_hot(y_int, self.label_shape[0])
