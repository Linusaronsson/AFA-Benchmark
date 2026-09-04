"""
Aggregate route structure into the paper's table and the figure's CSV.

Reads the per-namespace output of `route_redundancy.py`, keeps each dataset's
largest budget, and averages over the five dataset instances. Writes both
artefacts from one pass so `tab:route-structure` and `fig:structure` cannot
disagree.

The descriptors are method-independent: the search uses a masked classifier and
random feature subsets, and no AFA method appears in it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import pandas as pd

from afabench.plotting.methods import DATASET_LABELS_SHORT

COLUMNS = {
    "static_reference_score": "v_static",
    "route_sensitivity": "route_sensitivity",
    "weighted_route_overlap": "weighted_route_overlap",
}

HIGHLIGHT_DIRECTIONS = {
    "v_static": "min",
    "route_sensitivity": "max",
    "weighted_route_overlap": "max",
}

DEFAULT_NAMESPACES = (
    "core_group_missingness_v2",
    "induced_nonuniform_missingness_v2",
    "induced_real_missingness_v2",
)
PAPER_DATASETS = frozenset(
    {
        "actg",
        "cube",
        "cube_nm",
        "cube_nonuniform_costs",
        "diabetes",
        "heart_disease",
        "miniboone",
        "nhanes_mortality",
    }
)
_BASE_COLUMNS = frozenset({"dataset", "instance", "budget", *COLUMNS.keys()})
_PROTOCOL_COLUMNS = frozenset(
    {
        "selection_split",
        "eval_split",
        "metric",
        "k_requested",
        "k_unique",
        "static_reference_cost",
        "empty_route_selection_score",
        "random_route_score_mean",
    }
)


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    return cast("pd.Series", frame[name])


def _read_routes(paths: list[Path], *, protocol: bool) -> pd.DataFrame:
    missing = [path for path in paths if not path.exists()]
    if missing:
        message = f"missing route CSVs: {[str(path) for path in missing]}"
        raise FileNotFoundError(message)
    if not paths:
        message = "at least one route CSV is required"
        raise ValueError(message)
    required = _BASE_COLUMNS | (_PROTOCOL_COLUMNS if protocol else frozenset())
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        absent = sorted(required - set(frame.columns))
        if absent:
            msg = f"{path} is missing required columns: {absent}"
            raise ValueError(msg)
        frames.append(frame)
    routes = pd.concat(frames, ignore_index=True)
    cell_columns = ["dataset", "instance", "budget"]
    duplicates = routes.duplicated(cell_columns, keep=False)
    if duplicates.any():
        cells = routes.loc[duplicates, cell_columns].to_dict("records")
        msg = f"duplicate route cells: {cells}"
        raise ValueError(msg)
    return routes


def _validate_protocol(
    routes: pd.DataFrame,
    *,
    expected_k: int | None,
    selection_split: str | None,
    eval_split: str | None,
) -> None:
    if selection_split is not None and not bool(
        _column(routes, "selection_split").eq(selection_split).all()
    ):
        msg = f"all routes must use selection split {selection_split!r}"
        raise ValueError(msg)
    if eval_split is not None and not bool(
        _column(routes, "eval_split").eq(eval_split).all()
    ):
        msg = f"all routes must use evaluation split {eval_split!r}"
        raise ValueError(msg)
    if expected_k is not None and not bool(
        _column(routes, "k_requested").eq(expected_k).all()
    ):
        msg = f"all routes must request K={expected_k}"
        raise ValueError(msg)


def _validate_largest_routes(
    routes: pd.DataFrame,
    *,
    expected_datasets: frozenset[str] | None,
    expected_instances: int | None,
) -> None:
    datasets = frozenset(str(value) for value in routes["dataset"].unique())
    if expected_datasets is not None and datasets != expected_datasets:
        missing_datasets = sorted(expected_datasets - datasets)
        extra_datasets = sorted(datasets - expected_datasets)
        msg = (
            "largest-budget routes do not match the paper datasets: "
            f"missing={missing_datasets}, extra={extra_datasets}"
        )
        raise ValueError(msg)
    if expected_instances is not None:
        expected = set(range(expected_instances))
        invalid = {
            str(dataset): sorted({int(value) for value in group["instance"]})
            for dataset, group in routes.groupby("dataset")
            if {int(value) for value in group["instance"]} != expected
        }
        if invalid:
            msg = (
                "largest-budget routes must contain instances "
                f"{sorted(expected)} exactly once per dataset; got {invalid}"
            )
            raise ValueError(msg)
    overlap = cast(
        "pd.Series",
        pd.to_numeric(
            _column(routes, "weighted_route_overlap"), errors="coerce"
        ),
    )
    if bool(overlap.isna().any()) or not bool(overlap.between(0.0, 1.0).all()):
        msg = "largest-budget weighted route overlap must be finite and in [0, 1]"
        raise ValueError(msg)


def collect(
    paths: list[Path],
    *,
    expected_datasets: frozenset[str] | None = None,
    expected_instances: int | None = None,
    expected_k: int | None = None,
    selection_split: str | None = None,
    eval_split: str | None = None,
) -> pd.DataFrame:
    """Mean and standard error per dataset, at its largest budget."""
    protocol = any(
        value is not None
        for value in (
            expected_datasets,
            expected_instances,
            expected_k,
            selection_split,
            eval_split,
        )
    )
    routes = _read_routes(paths, protocol=protocol)
    _validate_protocol(
        routes,
        expected_k=expected_k,
        selection_split=selection_split,
        eval_split=eval_split,
    )
    largest = _column(routes, "budget") == routes.groupby("dataset")[
        "budget"
    ].transform("max")
    routes = cast("pd.DataFrame", routes[largest])

    _validate_largest_routes(
        routes,
        expected_datasets=expected_datasets,
        expected_instances=expected_instances,
    )

    rows = []
    for dataset, group in routes.groupby("dataset"):
        record: dict[str, object] = {
            "dataset": str(dataset),
            "budget": int(_column(group, "budget").iloc[0]),
            "instances": len(group),
        }
        for source, name in COLUMNS.items():
            values = _column(group, source)
            record[name] = float(values.mean())
            record[f"{name}_sem"] = float(values.sem())
        rows.append(record)
    return (
        pd.DataFrame(rows)
        .sort_values("weighted_route_overlap", ascending=False)
        .reset_index(drop=True)
    )


def _clear_extreme(
    frame: pd.DataFrame,
    name: str,
    direction: str,
) -> str | None:
    """Return a separated extreme under mean plus or minus two SE."""
    ordered = frame.dropna(subset=[name, f"{name}_sem"]).sort_values(
        name,
        ascending=direction == "min",
    )
    if len(ordered) < 2:
        return None
    extreme = ordered.iloc[0]
    runner_up = ordered.iloc[1]
    extreme_mean = float(extreme[name])
    extreme_sem = float(extreme[f"{name}_sem"])
    runner_mean = float(runner_up[name])
    runner_sem = float(runner_up[f"{name}_sem"])
    separated = (
        extreme_mean + 2 * extreme_sem < runner_mean - 2 * runner_sem
        if direction == "min"
        else extreme_mean - 2 * extreme_sem > runner_mean + 2 * runner_sem
    )
    return str(extreme["dataset"]) if separated else None


def _estimate_cell(
    record: dict[str, object],
    name: str,
    highlighted_dataset: str | None,
) -> str:
    mean = float(cast("Any", record[name]))
    sem = float(cast("Any", record[f"{name}_sem"]))
    if pd.isna(mean) or pd.isna(sem):
        return "--"
    estimate = f"{mean:.3f} \\pm {sem:.3f}"
    if str(record["dataset"]) == highlighted_dataset:
        return f"$\\boldsymbol{{{estimate}}}$"
    return f"${estimate}$"


def render(frame: pd.DataFrame) -> str:
    highlights = {
        name: _clear_extreme(frame, name, direction)
        for name, direction in HIGHLIGHT_DIRECTIONS.items()
    }
    lines = [
        "% Generated by scripts/analysis/route_structure_table.py.",
        "% Regenerate rather than editing by hand.",
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Fixed-route structure, ordered by decreasing "
        "$\\omega_{\\mathrm{route}}$.",
        "Mean $\\pm$ standard error over five dataset instances.",
        "Static references and route weights are selected on training data; "
        "$V_{\\mathrm{static}}$ and $\\Delta_{\\mathrm{route}}$ are reported "
        "on validation data. Bold marks a directionally relevant extreme "
        "whose mean $\\pm 2$ standard errors does not overlap the runner-up.}",
        "\\label{tab:route-structure}",
        "{\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Dataset & $b$ & $V_{\\mathrm{static}}$ & "
        "$\\Delta_{\\mathrm{route}}$ & $\\omega_{\\mathrm{route}}$ \\\\",
        "\\midrule",
    ]
    for record in frame.to_dict("records"):
        dataset = str(record["dataset"])
        label = DATASET_LABELS_SHORT.get(dataset, dataset)
        cells = " & ".join(
            _estimate_cell(record, name, highlights[name])
            for name in (
                "v_static",
                "route_sensitivity",
                "weighted_route_overlap",
            )
        )
        lines.append(f"{label} & {int(record['budget'])} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "}", "\\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path("extra/output/missing_data/analysis"),
    )
    parser.add_argument(
        "--namespace",
        action="append",
        default=None,
        help="Namespaces to pool. Defaults to the three induced matrices.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path(
            "extra/output/paper/experiments/results/route_structure.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "extra/output/paper/experiments/results/route_structure.tex"
        ),
    )
    arguments = parser.parse_args()

    paper_protocol = arguments.namespace is None
    namespaces = arguments.namespace or list(DEFAULT_NAMESPACES)
    paths = [
        arguments.analysis_root / f"route_redundancy_{namespace}_val.csv"
        for namespace in namespaces
    ]
    frame = collect(
        paths,
        expected_datasets=PAPER_DATASETS if paper_protocol else None,
        expected_instances=5 if paper_protocol else None,
        expected_k=2000 if paper_protocol else None,
        selection_split="train" if paper_protocol else None,
        eval_split="val" if paper_protocol else None,
    )

    arguments.csv_output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.csv_output, index=False)
    arguments.output.write_text(render(frame) + "\n")

    print(frame.round(3).to_string(index=False))
    print(f"\nwrote {arguments.csv_output} and {arguments.output}")


if __name__ == "__main__":
    main()
