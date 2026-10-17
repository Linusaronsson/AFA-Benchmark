"""Run a benchmark config in the launcher's selected Python environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from afabench.release.config_run import run_config
from workflow.run import snakemake_arguments

if __name__ == "__main__":
    sys.exit(run_config(snakemake_arguments(sys.argv[1:])))
