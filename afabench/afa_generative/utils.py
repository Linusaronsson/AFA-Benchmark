from torch import nn


def restore_parameters(model: nn.Module, best_model: nn.Module) -> None:
    """Move parameters from best model to current model."""
    for param, best_param in zip(
        model.parameters(), best_model.parameters(), strict=False
    ):
        param.data = best_param
