"""Where training scripts send their metrics: wandb or nowhere."""

from collections.abc import Mapping
from typing import Protocol, override

from wandb.sdk.wandb_run import Run


class MetricLogger(Protocol):
    def log(self, metrics: Mapping[str, object]) -> None: ...

    def finish(self) -> None: ...


class NullMetricLogger(MetricLogger):
    """Discards metrics. Used when the contract sets `use_wandb=false`."""

    @override
    def log(self, metrics: Mapping[str, object]) -> None:
        pass

    @override
    def finish(self) -> None:
        pass


class WandbMetricLogger(MetricLogger):
    def __init__(self, run: Run) -> None:
        self._run: Run = run

    @override
    def log(self, metrics: Mapping[str, object]) -> None:
        self._run.log(dict(metrics))

    @override
    def finish(self) -> None:
        self._run.finish()
