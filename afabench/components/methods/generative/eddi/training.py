"""Assemble the EDDI AFA method from the `pvae` pretrained model."""

from pathlib import Path

from afabench.components.methods.generative.eddi.afa_methods import (
    EDDIAFAMethod,
)
from afabench.components.methods.rl.odin.models import ODINPretrainingModel
from afabench.core.bundle_system.torch_bundle import TorchModelBundle
from afabench.training.inputs import TrainingInputs


def build_eddi_afa_method(
    inputs: TrainingInputs, *, classifier_bundle_path: Path | None
) -> EDDIAFAMethod:
    """
    EDDI needs no training of its own beyond the pretrained partial VAE.

    With `classifier_bundle_path` set, EDDI predicts with that external
    classifier; otherwise with the pretrained model's built-in classifier.
    """
    pretrained_model = inputs.pretrained_model(TorchModelBundle).model
    if not isinstance(pretrained_model, ODINPretrainingModel):
        msg = (
            "EDDI expects the pvae pretrained model "
            f"(ODINPretrainingModel), got {type(pretrained_model).__name__}."
        )
        raise TypeError(msg)
    train_dataset = inputs.train_dataset()
    unmasker = inputs.unmasker()
    return EDDIAFAMethod(
        sampler=pretrained_model.partial_vae,
        predictor=pretrained_model.classifier,
        num_classes=train_dataset.label_shape[-1],
        classifier_bundle_path=classifier_bundle_path,
        selection_costs=unmasker.get_selection_costs(
            train_dataset.get_feature_acquisition_costs()
        ),
        n_contexts=getattr(unmasker, "n_contexts", None),
    )
