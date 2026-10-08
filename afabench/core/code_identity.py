"""
The code a pipeline job runs: the commit of the afabench checkout.

Kept apart from `provenance` so that recording a job does not import torch.
"""

import subprocess
from pathlib import Path

# The work tree holding the running afabench package, wherever it runs from
AFABENCH_CHECKOUT = Path(__file__).resolve().parents[2]


def code_identity(checkout: Path) -> tuple[str | None, bool | None]:
    """Return the checkout's commit and whether tracked files changed."""
    commit = _git(checkout, "rev-parse", "HEAD")
    if commit is None:
        return None, None
    # Untracked files (outputs, data) do not make the code dirty
    status = _git(checkout, "status", "--porcelain", "--untracked-files=no")
    return commit, bool(status)


def _git(checkout: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return completed.stdout.strip()
