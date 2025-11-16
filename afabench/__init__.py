import sys
from pathlib import Path

SAVE_PATH = Path("result") / Path(sys.argv[0]).stem
__all__ = ["SAVE_PATH"]
