import torch
from torch import nn

from afabench.components.methods.discriminative.dime.afa_methods import (
    DIMEAFAMethod,
)
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
)


def test_tabular_gdfs_accepts_image_shaped_eval_inputs() -> None:
    method = GDFSAFAMethod(
        selector=nn.Linear(2 * 28 * 28, 49),
        predictor=nn.Linear(2 * 28 * 28, 10),
        device=torch.device("cpu"),
        modality="tabular",
        d_in=28 * 28,
        d_out=10,
        n_selections=49,
    )
    masked_features = torch.zeros((2, 1, 28, 28))
    feature_mask = torch.zeros((2, 1, 28, 28), dtype=torch.bool)

    prediction = method.predict(
        masked_features=masked_features,
        feature_mask=feature_mask,
        feature_shape=torch.Size((1, 28, 28)),
    )
    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=torch.zeros((2, 49), dtype=torch.bool),
        feature_shape=torch.Size((1, 28, 28)),
    )

    assert prediction.shape == (2, 10)
    assert action.shape == (2, 1)


def test_tabular_dime_accepts_image_shaped_eval_inputs() -> None:
    method = DIMEAFAMethod(
        value_network=nn.Linear(2 * 28 * 28, 49),
        predictor=nn.Linear(2 * 28 * 28, 10),
        device=torch.device("cpu"),
        modality="tabular",
        d_in=28 * 28,
        d_out=10,
        n_selections=49,
    )
    masked_features = torch.zeros((2, 1, 28, 28))
    feature_mask = torch.zeros((2, 1, 28, 28), dtype=torch.bool)

    prediction = method.predict(
        masked_features=masked_features,
        feature_mask=feature_mask,
        feature_shape=torch.Size((1, 28, 28)),
    )
    action = method.act(
        masked_features=masked_features,
        feature_mask=feature_mask,
        selection_mask=torch.zeros((2, 49), dtype=torch.bool),
        feature_shape=torch.Size((1, 28, 28)),
    )

    assert prediction.shape == (2, 10)
    assert action.shape == (2, 1)
