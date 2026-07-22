import torch

from afabench.missing_values.stepwise import restore_acquired_features


class _RecordingRestorer:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(
        self,
        masked_features: torch.Tensor,
        feature_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.calls.append((masked_features.clone(), feature_mask.clone()))
        return masked_features + torch.tensor(
            [100.0, 200.0, 300.0],
            device=masked_features.device,
        )


def test_stepwise_restoration_conditions_on_new_factual_values() -> None:
    restorer = _RecordingRestorer()
    restored = restore_acquired_features(
        features=torch.tensor([[1.0, 2.0, 0.0]]),
        feature_mask=torch.tensor([[True, False, False]]),
        new_feature_mask=torch.tensor([[True, True, True]]),
        source_availability=torch.tensor([[True, True, False]]),
        restoration_fn=restorer,
    )

    assert len(restorer.calls) == 1
    inputs, mask = restorer.calls[0]
    assert torch.equal(inputs, torch.tensor([[1.0, 2.0, 0.0]]))
    assert torch.equal(mask, torch.tensor([[True, True, False]]))
    assert torch.equal(restored, torch.tensor([[1.0, 2.0, 300.0]]))


def test_stepwise_restoration_persists_previous_draws() -> None:
    restorer = _RecordingRestorer()
    first = restore_acquired_features(
        features=torch.tensor([[1.0, 0.0, 0.0]]),
        feature_mask=torch.tensor([[True, False, False]]),
        new_feature_mask=torch.tensor([[True, True, False]]),
        source_availability=torch.tensor([[True, False, False]]),
        restoration_fn=restorer,
    )
    second = restore_acquired_features(
        features=first,
        feature_mask=torch.tensor([[True, True, False]]),
        new_feature_mask=torch.tensor([[True, True, True]]),
        source_availability=torch.tensor([[True, False, False]]),
        restoration_fn=restorer,
    )

    inputs, mask = restorer.calls[1]
    assert torch.equal(inputs, torch.tensor([[1.0, 200.0, 0.0]]))
    assert torch.equal(mask, torch.tensor([[True, True, False]]))
    assert torch.equal(second, torch.tensor([[1.0, 200.0, 300.0]]))


def test_stepwise_restoration_is_identity_without_model() -> None:
    features = torch.tensor([[1.0, 2.0]])
    restored = restore_acquired_features(
        features,
        torch.tensor([[False, False]]),
        torch.tensor([[True, False]]),
        torch.tensor([[False, True]]),
        None,
    )

    assert restored is features
