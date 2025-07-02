from pathlib import Path
from typing import Self, final, override

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn
from torchrl.modules import MLP
from afa_rl.shim2018.custom_types import Embedder, Embedding, EmbeddingClassifier
from afa_rl.utils import (
    get_feature_set,
    mask_data,
    shuffle_feature_set,
)
from common.custom_types import (
    AFAClassifier,
    AFAPredictFn,
    FeatureMask,
    MaskedFeatures,
    Features,
    Label,
    Logits,
)


@final
class ReadProcessEncoder(nn.Module):
    """A set encoder using RNNs, as described in the paper "Order Matters: Sequence to sequence for sets" (http://arxiv.org/abs/1511.06391)."""

    def __init__(
        self,
        set_element_size: int,  # size of each element in input set. Usually 1 (scalar sequence)
        output_size: int,  # size of output vector, one output per sequence
        reading_block_cells: tuple[int, ...] = (32, 32),
        writing_block_cells: tuple[int, ...] = (32, 32),
        memory_size: int = 16,  # each element in the sequence gets converted into a memory with this size
        processing_steps: int = 5,  # RNN processing steps. Paper shows that 5-10 is good.
        dropout: float = 0.0,
    ):
        super().__init__()
        self.set_element_size = set_element_size
        self.output_size = output_size
        self.reading_block_cells = reading_block_cells
        self.writing_block_cells = writing_block_cells
        self.memory_size = memory_size
        self.processing_steps = processing_steps
        self.dropout = dropout

        self.reading_block = MLP(
            in_features=set_element_size,
            out_features=memory_size,
            num_cells=reading_block_cells,
            activation_class=nn.ReLU,
            dropout=dropout,
        )

        self.process_lstm = nn.LSTMCell(memory_size, memory_size)

        # After the processing steps, the final memory is passed through a MLP to produce the output
        # The paper uses a pointer network but we only want a single feature vector
        self.write_block = MLP(
            in_features=2 * memory_size,
            out_features=output_size,
            num_cells=writing_block_cells,
            activation_class=nn.ReLU,
            dropout=dropout,
        )

        # Empty sets are represented by a learnable vector
        self.empty_set_vector = nn.Parameter(torch.zeros(output_size))

    @override
    def forward(self, input_set: Tensor, lengths: Tensor) -> Tensor:
        """Forward pass.

        Args:
            input_set: a tensor of shape (batch_size, set_size, element_size). The valid elements have to be first in the sequence.
            lengths: a tensor of shape (batch_size). Number of valid elements in each sequence.

        Returns:
            output: a tensor of shape (batch_size, output_size)

        """
        # We want to support empty sets as well, but these have to be handled separately, look at the end of the function
        original_batch_size = input_set.shape[0]
        nonempty_set_mask = lengths > 0

        # Now only treat the non-empty sets
        input_set = input_set[nonempty_set_mask]
        lengths = lengths[nonempty_set_mask]
        batch_size, set_size, _ = input_set.shape

        # Read: Map each set elements to a memory vector
        memories = self.reading_block(input_set)  # (batch_size, set_size, memory_size)

        # Initialize lstm state
        q_t = torch.zeros(batch_size, self.memory_size, device=input_set.device)
        c_t = torch.zeros(batch_size, self.memory_size, device=input_set.device)
        # Initial input
        r_t = torch.zeros(batch_size, self.memory_size, device=input_set.device)

        # Process: Iteratively refine state with attention over memories
        for _ in range(self.processing_steps):
            # LSTM update
            q_t, c_t = self.process_lstm(
                r_t, (q_t, c_t)
            )  # q_t: (batch_size, memory_size)

            # Attention between each memory vector and all memories, even the invalid ones, to produce logits
            e_t = torch.bmm(memories, q_t.unsqueeze(-1)).squeeze(
                -1
            )  # (batch_size, set_size)

            # Mask padding elements
            mask = torch.arange(set_size, device=input_set.device).repeat(
                batch_size, 1
            ) < lengths.unsqueeze(1).repeat(1, set_size)
            e_t[~mask] = float("-inf")

            # Compute attention weights
            a_t = torch.softmax(e_t, dim=-1)  # (batch_size, set_size)

            # Read from memory with attention
            r_t = torch.bmm(a_t.unsqueeze(1), memories).squeeze(
                1
            )  # (batch_size, memory_size)

        # Write: Transform final state to output
        output = self.write_block(
            torch.cat((q_t, r_t), dim=1)
        )  # (batch_size, output_size)

        # Empty sets are represented by a learnable vector
        complete_output = self.empty_set_vector.expand(original_batch_size, -1).clone()
        complete_output[nonempty_set_mask] = output

        return complete_output


@final
class Shim2018Embedder(Embedder):
    """Wrap a ReadProcessEmbedder to handle inputs consisting of features and their indices, using the representation described in "Joint Active Feature Acquisition and Classification with Variable-Size Set Encoding."""

    def __init__(self, encoder: ReadProcessEncoder):
        super().__init__()
        self.encoder = encoder

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Embedding:
        feature_set, lengths = get_feature_set(
            masked_features, feature_mask
        )  # (batch_size, n_features, state_size)

        # Shuffling set should improve training
        feature_set = shuffle_feature_set(feature_set, lengths)

        return self.encoder(feature_set, lengths)


@final
class Shim2018MLPClassifier(EmbeddingClassifier):
    def __init__(self, input_size: int, num_classes: int, num_cells: tuple[int, ...]):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_cells = num_cells

        self.mlp = MLP(
            input_size, num_classes, num_cells=num_cells, activation_class=nn.ReLU
        )

    @override
    def forward(self, embedding: Embedding) -> Logits:
        return self.mlp(embedding)


@final
class LitShim2018EmbedderClassifier(pl.LightningModule):
    """A module that combines the ShimEncoder with a classifier for pretraining."""

    def __init__(
        self,
        embedder: Shim2018Embedder,
        classifier: Shim2018MLPClassifier,
        class_probabilities: Float[Tensor, "n_classes"],
        min_masking_probability: float = 0.0,
        max_masking_probability: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["embedder", "classifier"])
        self.lr = lr
        self.embedder = embedder
        self.classifier = classifier
        self.class_weight = 1 / class_probabilities
        self.class_weight = self.class_weight / torch.sum(self.class_weight)
        self.min_masking_probability = min_masking_probability
        self.max_masking_probability = max_masking_probability

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> tuple[Embedding, Logits]:
        """Forward pass.

        Args:
            masked_features: currently observed features, with zeros for missing features
            feature_mask: indicator for missing features, 1 if feature is observed, 0 if missing

        Returns:
            embedding: the embedding of the input features
            classifier_output: the output of the classifier

        """
        embedding = self.embedder(masked_features, feature_mask)
        classifier_output = self.classifier(embedding)
        return embedding, classifier_output

    @override
    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        features: Features = batch[0]
        label: Label = batch[1]

        masking_probability = self.min_masking_probability + torch.rand(1).item() * (
            self.max_masking_probability - self.min_masking_probability
        )
        self.log("masking_probability", masking_probability, sync_dist=True)

        masked_features, feature_mask, _ = mask_data(features, p=masking_probability)
        _, y_hat = self(masked_features, feature_mask)
        loss = F.cross_entropy(y_hat, label, weight=self.class_weight.to(y_hat.device))
        self.log("train_loss", loss)
        return loss

    def _get_loss_and_acc(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask, y: Tensor
    ) -> tuple[Tensor, Tensor]:
        _, y_hat = self(masked_features, feature_mask)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weight.to(y_hat.device))
        predicted_class = torch.argmax(y_hat, dim=1)
        true_class = torch.argmax(y, dim=1)
        acc = (predicted_class == true_class).float().mean()
        return loss, acc

    @override
    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int):
        feature_values, y = batch

        # Mask features with minimum probability -> see many features (observations)
        feature_mask_many_observations = (
            torch.rand(feature_values.shape, device=feature_values.device)
            > self.min_masking_probability
        )
        feature_values_many_observations = feature_values.clone()
        feature_values_many_observations[feature_mask_many_observations == 0] = 0
        loss_many_observations, acc_many_observations = self._get_loss_and_acc(
            feature_values_many_observations, feature_mask_many_observations, y
        )
        self.log("val_loss_many_observations", loss_many_observations)
        self.log("val_acc_many_observations", acc_many_observations)

        # Mask features with maximum probability -> see few features (observations)
        feature_mask_few_observations = (
            torch.rand(feature_values.shape, device=feature_values.device)
            > self.max_masking_probability
        )
        feature_values_few_observations = feature_values.clone()
        feature_values_few_observations[feature_mask_few_observations == 0] = 0
        loss_few_observations, acc_few_observations = self._get_loss_and_acc(
            feature_values_few_observations, feature_mask_few_observations, y
        )
        self.log("val_loss_few_observations", loss_few_observations)
        self.log("val_acc_few_observations", acc_few_observations)

    @override
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@final
class Shim2018AFAPredictFn(AFAPredictFn):
    """A wrapper for the ShimEmbedderClassifier to make it compatible with the AFAPredictFn interface."""

    def __init__(self, embedder_and_classifier: LitShim2018EmbedderClassifier):
        super().__init__()
        self.embedder_and_classifier = embedder_and_classifier

    @override
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        _, logits = self.embedder_and_classifier(masked_features, feature_mask)
        return logits.softmax(dim=-1)


@final
class Shim2018AFAClassifier(AFAClassifier):
    """A wrapper for the ShimEmbedderClassifier to make it compatible with the AFAClassifier interface."""

    def __init__(
        self,
        embedder_and_classifier: LitShim2018EmbedderClassifier,
        device: torch.device,
    ):
        super().__init__()
        self._device = device
        self.embedder_and_classifier = embedder_and_classifier.to(self._device)

    @override
    def __call__(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> Label:
        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        _, logits = self.embedder_and_classifier(masked_features, feature_mask)
        return logits.softmax(dim=-1).to(original_device)

    @override
    def save(self, path: Path) -> None:
        torch.save(self.embedder_and_classifier.cpu(), path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        embedder_and_classifier = torch.load(
            path, weights_only=False, map_location=device
        )
        return cls(embedder_and_classifier, device)

    @override
    def to(self, device: torch.device) -> Self:
        self.embedder_and_classifier = self.embedder_and_classifier.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device
