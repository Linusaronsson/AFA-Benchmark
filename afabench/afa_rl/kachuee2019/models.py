from pathlib import Path
from typing import Self, final, override

import lightning as pl
import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from afabench.afa_rl.common.utils import mask_data
from afabench.common.config_classes import Kachuee2019PQModuleConfig
from afabench.common.custom_types import (
    AFAClassifier,
    AFAPredictFn,
    FeatureMask,
    Features,
    Label,
    MaskedFeatures,
)
from afabench.common.utils import flatten_afa_input


@final
class Kachuee2019PQModule(nn.Module):
    """The architecture proposed in the paper "Opportunistic Learning: Budgeted Cost-Sensitive Learning from Data Streams", slightly cleaned up from the implementation found at https://github.com/mkachuee/Opportunistic/blob/master/Demo_OL_DQN.ipynb."""

    def __init__(
        self, n_features: int, n_classes: int, cfg: Kachuee2019PQModuleConfig
    ):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.cfg = cfg

        # Create network
        self.layers_p = nn.ModuleList()
        self.layers_q = nn.ModuleList()

        # Initialize the P-Net
        size_last = self.n_features
        for n_h in self.cfg.n_hiddens + [self.n_classes]:
            self.layers_p.append(nn.Linear(size_last, n_h))
            size_last = n_h

        # Initialize the Q-Net. The inputs to the Q-network are concatenations of the previous Q-network layer's output and activations from the P-network.
        size_last = self.n_features
        # always share_pq
        self.n_hiddens_q = []
        for ind in range(len(self.cfg.n_hiddens)):
            if ind == 0:
                self.n_hiddens_q.append(self.cfg.n_hiddens[ind])
            else:
                n_h = (
                    self.cfg.n_hiddens[ind - 1] * self.cfg.n_hiddens[ind]
                ) // (self.cfg.n_hiddens[ind - 1] + self.n_hiddens_q[-1])
                self.n_hiddens_q.append(n_h)
        for n_h, n_h_q in zip(
            self.cfg.n_hiddens, self.n_hiddens_q, strict=False
        ):
            self.layers_q.append(nn.Linear(size_last, n_h_q))
            size_last = n_h + n_h_q

        # Output of Q-Net now also includes the stop action
        self.layers_q.append(nn.Linear(size_last, self.n_features + 1))

    @override
    def forward(
        self, masked_features: MaskedFeatures
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # P-Net forward path
        act_last = masked_features
        acts_p = []
        for f_layer in self.layers_p[:-1]:
            act_last = F.dropout(
                F.relu(f_layer(act_last)), p=self.cfg.p_dropout
            )
            acts_p.append(act_last)
        class_logits = self.layers_p[-1](act_last)

        # Q-Net forward path, gradients are not backpropagated to P-Net
        act_last = masked_features
        act_last = F.relu(self.layers_q[0](act_last))
        for f_layer, p_act in zip(
            self.layers_q[1:-1], acts_p[:-1], strict=False
        ):
            p_act = p_act.detach()  # noqa: PLW2901
            act_last = F.relu(f_layer(torch.cat([act_last, p_act], dim=1)))
        p_act = acts_p[-1].detach()
        qvalues = self.layers_q[-1](torch.cat([act_last, p_act], dim=1))

        return class_logits, qvalues

    def confidence(
        self, masked_features: MaskedFeatures, mcdrop_samples: int = 1
    ) -> torch.Tensor:
        """
        Calculate the confidence histogram for each class given a sample.

        Args:
            masked_features: input sample of shape (batch_size, n_features)
            mcdrop_samples: mc dropout samples to use.
        """
        x_rep = masked_features.unsqueeze(1).expand(
            -1, mcdrop_samples, -1
        )  # (batch_size, mcdrop_samples, n_features)
        x_rep = x_rep.reshape(
            x_rep.shape[0] * mcdrop_samples, x_rep.shape[-1]
        )  # (batch_size*mcdrop_samples, n_features)
        class_logits, _qvalues = self.forward(
            x_rep
        )  # class_logits.shape = (batch_size*mcdrop_samples, n_classes)
        class_logits = class_logits.view(
            masked_features.shape[0], mcdrop_samples, -1
        )  # (batch_size, mcdrop_samples, n_classes)
        class_probabilities = class_logits.softmax(
            dim=-1
        )  # (batch_size, mcdrop_samples, n_classes)
        return class_probabilities.mean(dim=1)  # (batch_size, n_classes)

    def predict(
        self, x: torch.Tensor, mcdrop_samples: int = 1
    ) -> torch.Tensor:
        """
        Make class prediction for a sample.

        Args:
            x: input sample
            mcdrop_samples: mc dropout samples to use.
        """
        conf = self.confidence(x, mcdrop_samples)
        pred = torch.argmax(conf)
        return pred


@final
class LitKachuee2019PQModule(pl.LightningModule):
    def __init__(
        self,
        pq_module: Kachuee2019PQModule,
        class_probabilities: Float[torch.Tensor, "n_classes"],  # noqa: F821
        n_feature_dims: int,
        min_masking_probability: float = 0.0,
        max_masking_probability: float = 1.0,
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pq_module"])
        self.pq_module = pq_module
        self.n_feature_dims = n_feature_dims
        class_weight = 1 / class_probabilities
        self.class_weight = class_weight / class_weight.sum()
        self.min_masking_probability = min_masking_probability
        self.max_masking_probability = max_masking_probability
        self.lr = lr

    @override
    def forward(
        self, masked_features: MaskedFeatures, feature_mask: FeatureMask
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            masked_features: currently observed features, with zeros for missing features
            feature_mask: indicator for missing features, 1 if feature is observed, 0 if missing

        Returns:
            logits: the logits from the PQ network
            qvalues: the Q-values from the PQ network

        """
        class_logits, qvalues = self.pq_module(masked_features)
        return class_logits, qvalues

    @override
    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        features: Features = batch[0]
        label: Label = batch[1]

        masking_probability = self.min_masking_probability + torch.rand(
            1
        ).item() * (
            self.max_masking_probability - self.min_masking_probability
        )
        self.log("masking_probability", masking_probability, sync_dist=True)

        masked_features, feature_mask, _ = mask_data(
            features, p=masking_probability
        )
        class_logits, _ = self.forward(masked_features, feature_mask)
        loss = F.cross_entropy(
            class_logits,
            label,
            weight=self.class_weight.to(class_logits.device),
        )
        self.log("train_loss", loss)
        return loss

    def _get_loss_and_acc(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        class_logits, _ = self(masked_features, feature_mask)
        loss = F.cross_entropy(
            class_logits, y, weight=self.class_weight.to(class_logits.device)
        )
        predicted_class = torch.argmax(class_logits, dim=1)
        true_class = torch.argmax(y, dim=1)
        acc = (predicted_class == true_class).float().mean()
        return loss, acc

    @override
    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        feature_values, y = batch

        # Mask features with minimum probability -> see many features (observations)
        feature_mask_many_observations = (
            torch.rand(feature_values.shape, device=feature_values.device)
            > self.min_masking_probability
        )
        feature_values_many_observations = feature_values.clone()
        feature_values_many_observations[
            feature_mask_many_observations == 0
        ] = 0
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
    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


@final
class Kachuee2019AFAPredictFn(AFAPredictFn):
    """A wrapper for Kachuee2019PQModule to make it compatible with the AFAPredictFn interface."""

    def __init__(self, pq_module: Kachuee2019PQModule):
        super().__init__()
        self.pq_module = pq_module

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        # The model assumes flat features
        assert feature_shape is not None

        masked_features, feature_mask, label = flatten_afa_input(
            masked_features, feature_mask, label, feature_shape
        )
        class_logits, _qvalues = self.pq_module.forward(masked_features)
        return class_logits.softmax(dim=-1)


@final
class Kachuee2019AFAClassifier(AFAClassifier):
    """A wrapper for Kachuee2019PQModule to make it compatible with the AFAClassifier interface."""

    def __init__(
        self,
        pq_module: Kachuee2019PQModule,
        device: torch.device,
    ):
        super().__init__()
        self._device = device
        self.pq_module = pq_module.to(self._device)

    @override
    def __call__(
        self,
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> Label:
        # The model assumes flat features
        assert feature_shape is not None

        original_device = masked_features.device

        masked_features = masked_features.to(self._device)
        feature_mask = feature_mask.to(self._device)

        masked_features, feature_mask, label = flatten_afa_input(
            masked_features, feature_mask, label, feature_shape
        )

        class_logits, _qvalues = self.pq_module(masked_features)
        return class_logits.softmax(dim=-1).to(original_device)

    @override
    def save(self, path: Path) -> None:
        torch.save(self.pq_module.cpu(), path)

    @classmethod
    @override
    def load(cls, path: Path, device: torch.device) -> Self:
        pq_module = torch.load(path, weights_only=False, map_location=device)
        return cls(pq_module, device)

    @override
    def to(self, device: torch.device) -> Self:
        self.pq_module = self.pq_module.to(device)
        return self

    @property
    @override
    def device(self) -> torch.device:
        return self._device
