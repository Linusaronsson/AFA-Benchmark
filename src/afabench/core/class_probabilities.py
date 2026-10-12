"""Explicit upstream conversion to evaluation's probability-only interface."""

import torch

from afabench.core.types import (
    AFAClassifierFn,
    AFAPredictFn,
    ClassifierOutputKind,
    ClassProbabilities,
    FeatureMask,
    Label,
    MaskedFeatures,
)


def as_class_probability_fn(
    classifier_fn: AFAClassifierFn, *, output_kind: ClassifierOutputKind
) -> AFAPredictFn:
    declared_kind = str(output_kind)
    if declared_kind not in ("logits", "probabilities"):
        msg = f"Expected output_kind 'logits' or 'probabilities', got {output_kind!r}"
        raise ValueError(msg)

    def predict(
        masked_features: MaskedFeatures,
        feature_mask: FeatureMask,
        label: Label | None = None,
        feature_shape: torch.Size | None = None,
    ) -> ClassProbabilities:
        output = classifier_fn(
            masked_features=masked_features,
            feature_mask=feature_mask,
            label=label,
            feature_shape=feature_shape,
        )
        if output_kind == "logits":
            return output.softmax(dim=-1)
        return output

    return predict
