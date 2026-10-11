"""
Independent L2M encoder from Kobayashi et al., arXiv:2510.12624.

The specification's attention layout, where query instances read only the
context set, is used in both fit stages and evaluation, rather than the
paper's training-only target points.
There are no positional embeddings or stop logits.
"""

from pathlib import Path
from typing import Self, override

import torch
from jaxtyping import Bool, Float
from torch import nn

type TaskFeatures = Float[torch.Tensor, "*tasks sequence n_features"]
type TaskMask = (
    Bool[torch.Tensor, "*tasks sequence n_features"]
    | Float[torch.Tensor, "*tasks sequence n_features"]
)
type TaskLabels = Float[torch.Tensor, "*tasks sequence n_classes"]
type ClassifierLogits = Float[torch.Tensor, "*tasks query_instances n_classes"]
type PolicyLogits = Float[torch.Tensor, "*tasks query_instances n_features"]


class L2MModel(nn.Module):
    """
    Joint encoder with public classifier and policy heads.

    Forward accepts a sequence, or a batch of task sequences, with the
    labelled context set first and query instances last. It returns logits
    for query instances only, whose labels are always zeroed to prevent
    label leakage.
    Float masks are supported for straight-through acquisition gradients.
    Embedding depth counts the input projection and residual linear layers.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        *,
        model_dim: int,
        embedding_depth: int,
        n_layers: int,
        n_heads: int,
        feedforward_dim: int,
    ) -> None:
        super().__init__()
        self.architecture: dict[str, int] = {
            "n_features": n_features,
            "n_classes": n_classes,
            "model_dim": model_dim,
            "embedding_depth": embedding_depth,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "feedforward_dim": feedforward_dim,
        }
        for name, value in self.architecture.items():
            minimum = 2 if name == "n_classes" else 1
            if value < minimum:
                msg = f"{name}={value} must be at least {minimum}"
                raise ValueError(msg)
        if model_dim % n_heads:
            msg = (
                f"model_dim={model_dim} must be divisible by n_heads={n_heads}"
            )
            raise ValueError(msg)
        self.n_features: int = n_features
        self.n_classes: int = n_classes
        self.input_projection: nn.Linear = nn.Linear(
            2 * n_features + n_classes, model_dim
        )
        self.embedding_layers: nn.ModuleList = nn.ModuleList(
            nn.Sequential(nn.Linear(model_dim, model_dim), nn.ReLU())
            for _ in range(embedding_depth - 1)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            model_dim,
            n_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, n_layers, enable_nested_tensor=False
        )
        self.classifier_head: nn.Linear = nn.Linear(model_dim, n_classes)
        self.policy_head: nn.Linear = nn.Linear(model_dim, n_features)

    @property
    def device(self) -> torch.device:
        return self.input_projection.weight.device

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": self.architecture,
                "state_dict": self.state_dict(),
            },
            path / "model.pt",
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(
            path / "model.pt", map_location=device, weights_only=True
        )
        model = cls(**checkpoint["architecture"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()

    @override
    def forward(
        self,
        features: TaskFeatures,
        mask: TaskMask,
        labels: TaskLabels,
        *,
        context_set_size: int,
    ) -> tuple[ClassifierLogits, PolicyLogits]:
        if (
            features.ndim not in (2, 3)
            or features.shape[-1] != self.n_features
        ):
            msg = (
                f"features shape {features.shape} must be a sequence or "
                f"batch of sequences with {self.n_features} features"
            )
            raise ValueError(msg)
        if mask.shape != features.shape:
            msg = (
                f"mask shape {mask.shape} must match "
                f"features shape {features.shape}"
            )
            raise ValueError(msg)
        expected_labels = (*features.shape[:-1], self.n_classes)
        if labels.shape != expected_labels:
            msg = f"labels shape {labels.shape} must be {expected_labels}"
            raise ValueError(msg)
        sequence_length = features.shape[-2]
        if not 1 <= context_set_size < sequence_length:
            msg = (
                f"context_set_size={context_set_size} must be between 1 and "
                f"sequence length minus one ({sequence_length - 1})"
            )
            raise ValueError(msg)
        unbatched = features.ndim == 2
        if unbatched:
            features = features.unsqueeze(0)
            mask = mask.unsqueeze(0)
            labels = labels.unsqueeze(0)
        encoded_labels = torch.cat(
            (
                labels[:, :context_set_size],
                torch.zeros_like(labels[:, context_set_size:]),
            ),
            dim=1,
        )
        tokens = torch.cat(
            (features * mask, mask.to(features.dtype), encoded_labels),
            dim=-1,
        )
        embedded = self.input_projection(tokens)
        for layer in self.embedding_layers:
            embedded = embedded + layer(embedded)
        sequence_length = features.shape[1]
        # Every instance can read the context set; no instance can read a
        # query instance. Residual connections retain each query instance's
        # own observation.
        attention_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=features.device,
        )
        attention_mask[:, :context_set_size] = False
        query_encodings = self.encoder(embedded, mask=attention_mask)[
            :, context_set_size:
        ]
        classifier_logits = self.classifier_head(query_encodings)
        policy_logits = self.policy_head(query_encodings)
        if unbatched:
            return classifier_logits.squeeze(0), policy_logits.squeeze(0)
        return classifier_logits, policy_logits
