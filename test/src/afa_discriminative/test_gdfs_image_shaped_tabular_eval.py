import torch
from torch import nn

from afabench.components.methods.discriminative.common.checkpoint import (
    NetworkPairSpec,
)
from afabench.components.methods.discriminative.dime.afa_methods import (
    DIMEAFAMethod,
)
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSAFAMethod,
)

SPEC = NetworkPairSpec(
    modality="tabular",
    d_in=28 * 28,
    d_out=10,
    n_selections=49,
    policy_hidden_layers=[],
    classifier_hidden_layers=[],
    dropout=0.0,
)


def test_tabular_gdfs_accepts_image_shaped_eval_inputs() -> None:
    method = GDFSAFAMethod(
        nn.Linear(2 * 28 * 28, 49),
        nn.Linear(2 * 28 * 28, 10),
        SPEC,
        torch.device("cpu"),
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
        nn.Linear(2 * 28 * 28, 49),
        nn.Linear(2 * 28 * 28, 10),
        SPEC,
        torch.device("cpu"),
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
