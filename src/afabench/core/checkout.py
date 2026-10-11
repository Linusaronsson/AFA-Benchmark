"""
The afabench checkout the running package belongs to, if any.

Installed from a wheel, the package belongs to no checkout, and the
directories above it may be another project's repository.
"""

import tomllib
from pathlib import Path

PROJECT_NAME = "afa-benchmark"


def find_checkout(package: Path) -> Path | None:
    """Return the checkout holding the `package` directory, or None."""
    root = package.parent.parent
    if package.parent.name != "src":
        return None
    try:
        with (root / "pyproject.toml").open("rb") as f:
            name = tomllib.load(f).get("project", {}).get("name")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return root if name == PROJECT_NAME else None


# The work tree holding the running afabench package, wherever it runs from
AFABENCH_CHECKOUT = find_checkout(Path(__file__).resolve().parents[1])


def require_checkout(purpose: str) -> Path:
    """Return the checkout, or raise naming the `purpose` that needs it."""
    if AFABENCH_CHECKOUT is None:
        msg = (
            f"afabench is not running from a checkout, which {purpose} need;"
            " run it from a clone of the repository"
        )
        raise RuntimeError(msg)
    return AFABENCH_CHECKOUT
