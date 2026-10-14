"""
Prepare a tabular dataset's source from its provider's raw files (ADR 0014).

Writes the file dataset generation reads to the path the dataset's
generation config gives, relative to the repository root, and fails unless it has the SHA-256 the dataset
class pins. Committed sources need no preparation; run this to check one
against its provider or to restore it.

Usage:
    uv run python scripts/dataset_generation/prepare_source.py miniboone
"""

from pathlib import Path
from typing import Annotated

import typer
import yaml

from afabench.core.registry import get_class
from afabench.datasets.tabular_csv import TabularCSVDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "conf/scripts/dataset_generation/dataset"

app = typer.Typer(add_completion=False)


def source_of(
    dataset_key: str,
) -> tuple[type[TabularCSVDataset], Path, str | None]:
    """Return the dataset class of `dataset_key`, and the source path and SHA-256 its generation config gives."""
    config_file = CONFIG_DIR / f"{dataset_key}.yaml"
    if not config_file.exists():
        msg = f"No dataset generation config for {dataset_key!r} at {config_file}"
        raise FileNotFoundError(msg)
    config = yaml.safe_load(config_file.read_text())
    dataset_class = get_class(config["class_name"])
    if not issubclass(dataset_class, TabularCSVDataset):
        msg = (
            f"Dataset {dataset_key!r} is {config['class_name']}, which reads "
            "no tabular source"
        )
        raise TypeError(msg)
    kwargs = config["kwargs"]
    return (
        dataset_class,
        REPO_ROOT / kwargs[dataset_class.source_config_key],
        kwargs.get("sha256"),
    )


@app.command()
def prepare_source(
    dataset_key: Annotated[
        str, typer.Argument(help="Dataset key, such as miniboone.")
    ],
) -> None:
    """Download the provider's raw files and write the dataset's prepared source."""
    dataset_class, path, sha256 = source_of(dataset_key)
    dataset_class.prepare_source(path, sha256)
    typer.echo(f"Wrote {path}")


if __name__ == "__main__":
    app()
