import sys
from pathlib import Path

PLOT_PATH = Path("result")
SAVE_PATH = (
    Path("extra")
    / "result"
    / Path(sys.argv[0]).stem
    / Path(sys.argv[0]).parent.name
)
SAVE_PATH.mkdir(parents=True, exist_ok=True)
__all__ = ["SAVE_PATH", "PLOT_PATH"]
