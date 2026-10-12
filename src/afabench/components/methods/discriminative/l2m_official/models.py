"""
The authors' L2M model, `LTM`, wrapped for bundles and one-hot labels.

Upstream encodes binary labels as one column with a sigmoid classifier
(`label_dim: 1`, as in its MiniBooNE config) and labels with more classes
as one-hot columns with a softmax classifier. This benchmark's labels are
always one-hot, so the wrapper converts labels on the way in and classifier
outputs on the way out.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Self, override

import torch
from jaxtyping import Float
from torch import nn

from afabench.components.methods.discriminative.l2m_official.upstream.model_utils import (
    get_model,
)

if TYPE_CHECKING:
    from afabench.components.methods.discriminative.l2m_official.upstream.ltm import (
        LTM,
    )

type OneHotLabels = Float[torch.Tensor, "*batch n_classes"]
type UpstreamLabels = Float[torch.Tensor, "*batch label_dim"]
type ClassifierOutput = Float[torch.Tensor, "*batch label_dim"]
type ClassifierLogits = Float[torch.Tensor, "*batch n_classes"]

# A batch dictionary in upstream's layout: xc, yc, rc, xt, yt, rt, maskt.
type UpstreamBatch = dict[str, torch.Tensor]
type ModelConfig = dict[str, int | float | str | bool]


def label_dim(n_classes: int) -> int:
    """Upstream's label width: one column for binary labels."""
    if n_classes < 2:
        msg = f"n_classes={n_classes} must be at least 2"
        raise ValueError(msg)
    return 1 if n_classes == 2 else n_classes


class L2MOfficialModel(nn.Module):
    """
    Upstream `LTM` with its config, for both fit stages and evaluation.

    `model_config` holds the keys upstream's `get_model` reads.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__()
        self.model_config: ModelConfig = dict(model_config)
        self.ltm: LTM = get_model(self.model_config)

    @classmethod
    def build(
        cls,
        n_features: int,
        n_classes: int,
        *,
        model_dim: int,
        embedding_depth: int,
        prediction_depth: int,
        n_layers: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> Self:
        return cls(
            {
                "feature_dim": n_features,
                "label_dim": label_dim(n_classes),
                "n_classes": n_classes,
                "d_model": model_dim,
                "emb_depth": embedding_depth,
                "pred_depth": prediction_depth,
                "nhead": n_heads,
                "dim_feedforward": feedforward_dim,
                "num_layers": n_layers,
                "dropout": dropout,
                "task": "classification",
                # Read by the regression head only.
                "bound_std": True,
            }
        )

    @property
    def n_features(self) -> int:
        return int(self.model_config["feature_dim"])

    @property
    def n_classes(self) -> int:
        return int(self.model_config["n_classes"])

    @property
    def label_dim(self) -> int:
        return int(self.model_config["label_dim"])

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_config": self.model_config,
                "state_dict": self.ltm.state_dict(),
            },
            path / "model.pt",
        )

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        checkpoint = torch.load(
            path / "model.pt", map_location=device, weights_only=True
        )
        model = cls(checkpoint["model_config"]).to(device)
        model.ltm.load_state_dict(checkpoint["state_dict"])
        return model.eval()

    def upstream_labels(self, labels: OneHotLabels) -> UpstreamLabels:
        """One-hot labels in upstream's encoding: class 1's column if binary."""
        if labels.shape[-1] != self.n_classes:
            msg = (
                f"labels shape {tuple(labels.shape)} must end in "
                f"{self.n_classes} classes"
            )
            raise ValueError(msg)
        return labels[..., 1:] if self.label_dim == 1 else labels

    def classifier_logits(self, output: ClassifierOutput) -> ClassifierLogits:
        """
        Class logits from upstream's classifier output.

        A binary output `z` is the logit of class 1, so `[0, z]` has the
        same softmax as upstream's sigmoid.
        """
        if self.label_dim == 1:
            return torch.cat((torch.zeros_like(output), output), dim=-1)
        return output

    @override
    def forward(self, batch: UpstreamBatch) -> ClassifierLogits:
        """Class logits of the query instances, which read no labels."""
        query_batch = {**batch, "yt": torch.zeros_like(batch["yt"])}
        output = self.ltm.predictor(
            self.ltm.encode(query_batch, autoreg=False)
        )
        return self.classifier_logits(output)
