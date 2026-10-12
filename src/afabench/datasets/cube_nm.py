from typing import ClassVar, final, override

import torch

from afabench.core.types import Features, Label
from afabench.datasets.cube import CUBE_BINARY_CODES
from afabench.datasets.in_memory import InMemoryAFADataset
from afabench.datasets.utils import one_hot


@final
class CubeNMDataset(InMemoryAFADataset):
    """
    Synthetic tabular benchmark implementing the paper's CUBE-NM setup.

    Each sample has a logical context variable that selects which one of
    `n_contexts` CUBE-style 10-feature blocks is informative. The logical
    context is encoded as a noisy one-hot vector in the first `n_contexts`
    feature positions, matching the paper's implementation detail that the
    whole context can be acquired with one grouped action.

    Feature layout:
    - context encoding: `n_contexts` noisy one-hot features
    - block features: `n_contexts` blocks of size `block_size`
      Only the block chosen by the context contains class signal.

    Labels are 8-way one-hot vectors derived from the standard CUBE binary
    code pattern. The canonical benchmark setting uses `n_contexts=5`, which
    gives 55 observed features and 51 acquisition groups under
    `CubeNMUnmasker`.
    """

    config_keys: ClassVar[tuple[str, ...]] = (
        "n_samples",
        "seed",
        "context_feature_std",
        "informative_feature_std",
        "non_informative_feature_mean",
        "non_informative_feature_std",
        "n_contexts",
        "use_cheap_context_features",
    )
    block_size: ClassVar[int] = 10

    @classmethod
    @override
    def accepts_seed(cls) -> bool:
        return True

    @property
    @override
    def feature_shape(self) -> torch.Size:
        return torch.Size((self.n_features,))

    @property
    def n_features(self) -> int:
        return self.n_contexts + self.n_contexts * self.block_size

    @property
    @override
    def label_shape(self) -> torch.Size:
        return torch.Size([8])

    def __init__(
        self,
        n_samples: int = 20000,
        seed: int = 123,
        context_feature_std: float = 0.1,
        informative_feature_std: float = 0.1,
        non_informative_feature_mean: float = 0.5,
        non_informative_feature_std: float = 0.3,
        n_contexts: int = 5,
        *,
        use_cheap_context_features: bool = True,
    ):
        self.n_samples = n_samples
        self.seed = seed
        self.n_contexts = n_contexts
        self.context_feature_std = context_feature_std
        self.informative_feature_std = informative_feature_std
        self.non_informative_feature_mean = non_informative_feature_mean
        self.non_informative_feature_std = non_informative_feature_std
        self.use_cheap_context_features = use_cheap_context_features
        super().__init__()
        self.feature_costs = self._compute_feature_costs()

    @override
    def _build(self) -> tuple[Features, Label]:
        rng = torch.Generator().manual_seed(self.seed)
        # Sample the logical context, then encode it as a noisy one-hot
        # vector to match the benchmark setup used in the paper.
        context = torch.randint(
            0, self.n_contexts, (self.n_samples,), generator=rng
        )
        context_onehot = torch.nn.functional.one_hot(
            context, num_classes=self.n_contexts
        ).float() + torch.normal(
            mean=0,
            std=self.context_feature_std,
            size=(self.n_samples, self.n_contexts),
            generator=rng,
        )  # (n_samples, n_contexts)

        # Sample labels 0-7
        y_int = torch.randint(
            0,
            self.label_shape[0],
            (self.n_samples,),
            generator=rng,
        )

        # Create one candidate CUBE block per context value.
        blocks = []
        for _block_context in range(self.n_contexts):
            block = torch.normal(
                mean=self.non_informative_feature_mean,
                std=self.non_informative_feature_std,
                size=(self.n_samples, self.block_size),
                generator=rng,
            )
            blocks.append(block)
        blocks = torch.stack(blocks, dim=1)  # (n_samples, n_contexts, 10)

        # Only the context-selected block carries label information.
        for i in range(self.n_samples):
            ctx = int(context[i].item())
            label = int(y_int[i].item())
            bin_code = CUBE_BINARY_CODES[label].float()

            # Select 3 indices in the block to hold the binary code (arbitrary positions)
            insert_idx = [(label + j) % self.block_size for j in range(3)]

            noise = torch.normal(
                mean=0.0,
                std=self.informative_feature_std,
                size=(3,),
                generator=rng,
            )
            blocks[i, ctx, insert_idx] = bin_code + noise

        # Flatten blocks: (n_samples, n_contexts * block_size)
        block_features = blocks.view(self.n_samples, -1)

        # Final matrix: one-hot context features followed by all blocks.
        features = torch.cat(
            [context_onehot, block_features], dim=1
        )  # (n_samples, n_contexts + n_contexts * block_size)
        return features, one_hot(y_int, self.label_shape[0])

    def _compute_feature_costs(self) -> torch.Tensor:
        context_feature_cost_scaling = (
            1 / self.n_contexts if self.use_cheap_context_features else 1
        )
        context_feature_costs = context_feature_cost_scaling * torch.ones(
            self.n_contexts
        )
        remaining_feature_costs = torch.ones(self.n_features - self.n_contexts)
        all_feature_costs = torch.cat(
            [context_feature_costs, remaining_feature_costs], dim=-1
        )
        return all_feature_costs
