"""
NetworkPairTrainer: the training loop GDFS and DIME share.

Every case trains a handful of instances for a few epochs, so a case is a
test of the loop's control flow, not of what the methods learn: which
epoch's networks are kept, when a round stops, what is logged, and the
one DIME quirk the loop preserves on purpose.
"""

from collections.abc import Iterator
from typing import override

import pytest
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.optimizer import register_optimizer_step_pre_hook
from torch.utils.data import DataLoader, TensorDataset

from afabench.components.initializers.zero_initializer import ZeroInitializer
from afabench.components.methods.discriminative.common.trainer import (
    NetworkPairTrainer,
    TrainerSettings,
    TrainingObjective,
)
from afabench.components.methods.discriminative.common.utils import (
    MaskLayer,
    tie_first_k_linears_by_module,
)
from afabench.components.methods.discriminative.dime.afa_methods import (
    DIMETrainingObjective,
)
from afabench.components.methods.discriminative.gdfs.afa_methods import (
    GDFSTrainingObjective,
)
from afabench.components.unmaskers.direct_unmasker import DirectUnmasker

N_INSTANCES = 6
N_FEATURES = 4
N_CLASSES = 3
SELECTIONS_PER_EPISODE = 3
GDFS_METRICS = {
    "epoch",
    "temperature",
    "train_loss",
    "val_loss",
    "val_hard_loss",
}
DIME_METRICS = {
    "epoch",
    "train_value_loss",
    "train_predictor_loss",
    "val_loss_mean",
    "val_perf_mean",
    "val_loss_final",
    "val_perf_final",
    "epsilon",
}


class _ScriptedValidationLoss(nn.Module):
    """Answer each epoch's scripted score whatever the predictions, so the test chooses the best epoch."""

    def __init__(self, scores_per_epoch: list[float], calls_per_epoch: int):
        super().__init__()
        self.scores_per_epoch: list[float] = scores_per_epoch
        self.calls_per_epoch: int = calls_per_epoch
        self.calls: int = 0

    @override
    def forward(self, pred: Tensor, y: Tensor) -> Tensor:
        del pred, y
        score = self.scores_per_epoch[self.calls // self.calls_per_epoch]
        self.calls += 1
        return torch.tensor(score)


def _networks() -> tuple[nn.Module, nn.Module]:
    torch.manual_seed(7)
    policy_network = nn.Sequential(
        nn.Linear(2 * N_FEATURES, 16), nn.ReLU(), nn.Linear(16, N_FEATURES)
    )
    predictor = nn.Sequential(
        nn.Linear(2 * N_FEATURES, 16), nn.ReLU(), nn.Linear(16, N_CLASSES)
    )
    return policy_network, predictor


def _loader() -> DataLoader[tuple[Tensor, ...]]:
    torch.manual_seed(1)
    features = torch.randn(N_INSTANCES, N_FEATURES)
    label = torch.randint(N_CLASSES, (N_INSTANCES,))
    return DataLoader(
        TensorDataset(features, label), batch_size=N_INSTANCES, shuffle=False
    )


def _trainer(
    objective: TrainingObjective,
) -> tuple[NetworkPairTrainer, nn.Module, nn.Module]:
    policy_network, predictor = _networks()
    trainer = NetworkPairTrainer(
        policy_network=policy_network,
        predictor=predictor,
        mask_layer=MaskLayer(append=True),
        initializer=ZeroInitializer(),
        unmasker=DirectUnmasker(),
        objective=objective,
    )
    return trainer, policy_network, predictor


def _settings(
    objective: TrainingObjective,
    nepochs: int,
    val_loss_fn: nn.Module | None = None,
) -> TrainerSettings:
    # GDFS backpropagates the batch's loss per selection, DIME regresses
    # on each instance's loss drop, so each wants its own reduction
    loss_fn = (
        nn.CrossEntropyLoss()
        if objective.name == "gdfs"
        else nn.CrossEntropyLoss(reduction="none")
    )
    return TrainerSettings(
        lr=1e-2,
        nepochs=nepochs,
        selections_per_episode=SELECTIONS_PER_EPISODE,
        loss_fn=loss_fn,
        val_loss_fn=val_loss_fn,
        val_loss_mode=None if val_loss_fn is None else "min",
        min_lr=1e-6,
        patience=0,
        early_stopping_epochs=1,
        verbose=False,
    )


def _objectives() -> Iterator[tuple[str, TrainingObjective, int]]:
    """Each method's objective with how often it scores one validation epoch."""
    yield "gdfs", GDFSTrainingObjective(temp_steps=1), 2
    yield "dime", DIMETrainingObjective(eps=0.1), SELECTIONS_PER_EPISODE + 1


@pytest.mark.parametrize(
    ("objective", "expected_keys"),
    [
        (GDFSTrainingObjective(temp_steps=1), GDFS_METRICS),
        (DIMETrainingObjective(eps=0.1), DIME_METRICS),
    ],
    ids=["gdfs", "dime"],
)
def test_fit_logs_each_methods_metrics_once_per_epoch(
    objective: TrainingObjective, expected_keys: set[str]
) -> None:
    trainer, _, _ = _trainer(objective)
    logged: list[dict[str, float]] = []

    trainer.fit(
        _loader(),
        _loader(),
        _settings(objective, nepochs=2),
        metric_logger=logged.append,
    )

    assert [entry[f"{objective.name}/epoch"] for entry in logged] == [1.0, 2.0]
    assert all(
        set(entry) == {f"{objective.name}/{key}" for key in expected_keys}
        for entry in logged
    )


@pytest.mark.parametrize(
    ("objective", "calls_per_epoch"),
    [(o, calls) for _, o, calls in _objectives()],
    ids=[name for name, _, _ in _objectives()],
)
def test_fit_keeps_the_networks_of_the_best_validation_epoch_and_stops_after_the_plateau(
    objective: TrainingObjective, calls_per_epoch: int
) -> None:
    trainer, policy_network, predictor = _trainer(objective)
    # Epoch 2 is best; epochs 3 and 4 are bad, one more than the
    # early_stopping_epochs of 1, so the loop stops after epoch 4
    val_loss_fn = _ScriptedValidationLoss(
        [1.0, 0.5, 2.0, 2.0, 2.0, 2.0], calls_per_epoch
    )
    snapshots: list[list[Tensor]] = []

    def snapshot(_metrics: dict[str, float]) -> None:
        snapshots.append(
            [
                p.detach().clone()
                for p in [
                    *policy_network.parameters(),
                    *predictor.parameters(),
                ]
            ]
        )

    trainer.fit(
        _loader(),
        _loader(),
        _settings(objective, nepochs=6, val_loss_fn=val_loss_fn),
        metric_logger=snapshot,
    )

    assert len(snapshots) == 4
    kept = [*policy_network.parameters(), *predictor.parameters()]
    for parameter, best in zip(kept, snapshots[1], strict=True):
        torch.testing.assert_close(parameter, best)
    assert any(
        not torch.equal(parameter, last)
        for parameter, last in zip(kept, snapshots[-1], strict=True)
    )


def test_gdfs_trains_one_round_per_temperature_and_counts_epochs_across_rounds() -> (
    None
):
    objective = GDFSTrainingObjective(
        start_temp=1.0, end_temp=0.1, temp_steps=2
    )
    trainer, _, _ = _trainer(objective)
    val_loss_fn = _ScriptedValidationLoss(
        [1.0, 2.0, 2.0, 1.0, 2.0, 2.0], calls_per_epoch=2
    )
    logged: list[dict[str, float]] = []

    trainer.fit(
        _loader(),
        _loader(),
        _settings(objective, nepochs=3, val_loss_fn=val_loss_fn),
        metric_logger=logged.append,
    )

    assert [entry["gdfs/temperature"] for entry in logged] == pytest.approx(
        [1.0, 1.0, 1.0, 0.1, 0.1, 0.1]
    )
    assert [entry["gdfs/epoch"] for entry in logged] == [1, 2, 3, 4, 5, 6]


def test_dime_decays_epsilon_on_a_plateau_and_stops_after_eps_steps_decays() -> (
    None
):
    objective = DIMETrainingObjective(eps=0.5, eps_decay=0.2, eps_steps=2)
    trainer, _, _ = _trainer(objective)
    val_loss_fn = _ScriptedValidationLoss(
        [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
        calls_per_epoch=SELECTIONS_PER_EPISODE + 1,
    )
    logged: list[dict[str, float]] = []

    trainer.fit(
        _loader(),
        _loader(),
        _settings(objective, nepochs=8, val_loss_fn=val_loss_fn),
        metric_logger=logged.append,
    )

    # Epochs 2 and 3 are bad: decay once and go on; epochs 4 and 5 are bad
    # again: decay a second time and stop
    assert [entry["dime/epsilon"] for entry in logged] == pytest.approx(
        [0.5, 0.5, 0.5, 0.1, 0.1]
    )


def test_a_validation_metric_needs_its_mode() -> None:
    objective = DIMETrainingObjective(eps=0.1)
    trainer, _, _ = _trainer(objective)
    settings = TrainerSettings(
        lr=1e-2,
        nepochs=1,
        selections_per_episode=SELECTIONS_PER_EPISODE,
        loss_fn=nn.CrossEntropyLoss(reduction="none"),
        val_loss_fn=nn.CrossEntropyLoss(),
        val_loss_mode=None,
        min_lr=1e-6,
    )

    with pytest.raises(ValueError, match="val_loss_mode"):
        trainer.fit(_loader(), _loader(), settings)


def test_dime_lists_its_tied_parameters_twice_in_the_optimizer() -> None:
    """
    DIME ties the value network's first linear layers to the predictor's.

    Upstream DIME deduplicates them (`Adam(self.parameters())`); this port
    has always listed both networks' parameters, so a tied parameter is
    stepped twice per update. The loop keeps that on purpose until the
    maintainers decide otherwise (`docs/explanation/gdfs_dime_shared_code.md`).
    """
    objective = DIMETrainingObjective(eps=0.0)
    trainer, policy_network, predictor = _trainer(objective)
    tie_first_k_linears_by_module(predictor, policy_network, k=1)
    tied = next(predictor.parameters())
    assert tied is next(policy_network.parameters())
    listed: list[int] = []

    def count_tied(
        optimizer: Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        listed.append(
            sum(
                parameter is tied
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
        )

    handle = register_optimizer_step_pre_hook(count_tied)
    try:
        with pytest.warns(UserWarning, match="duplicate parameters"):
            trainer.fit(_loader(), _loader(), _settings(objective, nepochs=1))
    finally:
        handle.remove()

    assert listed == [2]
