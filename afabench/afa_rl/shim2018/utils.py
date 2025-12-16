import torch
from jaxtyping import Float
from torch import Tensor

from afabench.afa_rl.shim2018.models import (
    LitShim2018EmbedderClassifier,
    ReadProcessEncoder,
    Shim2018Embedder,
    Shim2018MLPClassifier,
)
from afabench.common.config_classes import Shim2018PretrainConfig


def get_shim2018_model_from_config(
    cfg: Shim2018PretrainConfig,
    feature_shape: torch.Size,
    n_classes: int,
    class_probabilities: Float[Tensor, "n_classes"],  # noqa: F821
) -> LitShim2018EmbedderClassifier:
    encoder = ReadProcessEncoder(
        set_element_size=feature_shape.numel()
        + 1,  # state contains one value and one index
        output_size=cfg.encoder.output_size,
        reading_block_cells=tuple(cfg.encoder.reading_block_cells),
        writing_block_cells=tuple(cfg.encoder.writing_block_cells),
        memory_size=cfg.encoder.memory_size,
        processing_steps=cfg.encoder.processing_steps,
        dropout=cfg.encoder.dropout,
    )
    embedder = Shim2018Embedder(encoder)
    classifier = Shim2018MLPClassifier(
        cfg.encoder.output_size, n_classes, tuple(cfg.classifier.num_cells)
    )
    lit_model = LitShim2018EmbedderClassifier(
        embedder=embedder,
        classifier=classifier,
        class_probabilities=class_probabilities,
        min_masking_probability=cfg.min_masking_probability,
        max_masking_probability=cfg.max_masking_probability,
        lr=cfg.lr,
    )
    return lit_model
