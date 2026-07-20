"""
Minimal phase timing: accumulate wall time per named phase, dump to JSON.

Deliberately tiny. No sampling, no nesting, no configuration. It exists to
separate bundle loading from rollout from saving, because the Snakemake-level
timers measure the whole subprocess and cannot tell those apart.

Fixed startup cost (interpreter, imports, CUDA context, Hydra composition) is
deliberately not measured here: capturing it would mean hoisting a timestamp
above the import block and suppressing E402 on every import. Derive it instead
as the residual of Snakemake's `benchmark:` wall time minus the phases below.
"""

import json
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch

_phases: defaultdict[str, float] = defaultdict(float)


def _sync() -> None:
    # CUDA kernels are async. Without this, a phase's GPU work is billed to
    # whichever later phase happens to synchronize first.
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Accumulate wall time spent in this block under `name`."""
    _sync()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        _phases[name] += time.perf_counter() - start


def dump(path: str | Path) -> None:
    """Write accumulated phases to JSON. Sidecar file, not a pipeline output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(_phases), indent=2, sort_keys=True))
