"""
Collect every job record under an output root into the job duration table.

Run by the pipeline's `collect_job_records` rule; see
`afabench.core.job_duration_table` and `docs/reference/job_records.md`.

Usage:
    python scripts/misc/collect_job_records.py \
        --output-root extra/output/production \
        --output extra/output/production/merged_results/job_duration_table.parquet
"""

from pathlib import Path
from typing import Annotated

import typer

from afabench.core.job_duration_table import write_job_duration_table

app = typer.Typer(add_completion=False)


@app.command()
def collect_job_records(
    output_root: Annotated[
        Path, typer.Option(help="Output root whose job records to collect.")
    ],
    output: Annotated[
        Path, typer.Option(help="Parquet file to write the table to.")
    ],
) -> None:
    """Write one table row per job record, failed and timed-out included."""
    write_job_duration_table(output_root, output)


if __name__ == "__main__":
    app()
