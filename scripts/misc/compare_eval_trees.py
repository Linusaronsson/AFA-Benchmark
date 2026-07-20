"""
Compare two trees of eval_data.parquet files.

This is the contract for the optimization work: any change meant to preserve
results must leave every eval_data.parquet identical. Anything that does differ
has to be explained and quantified, not waved through, so mismatches are
reported per column with a row count rather than as a bare boolean.

    uv run python scripts/misc/compare_eval_trees.py BASELINE_DIR CURRENT_DIR

Exits 0 when every shared file matches, 1 otherwise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _relative_parquets(root: Path) -> dict[Path, Path]:
    return {
        p.relative_to(root): p for p in sorted(root.rglob("eval_data.parquet"))
    }


def _comparable(s: pd.Series) -> pd.Series:
    """
    Make a column element-wise comparable.

    `prev_selections_performed` holds a variable-length array per row, and
    elementwise `!=` on those raises rather than returning a mask, so collapse
    them to tuples first.
    """
    if s.dtype == object:
        return s.map(
            lambda v: tuple(v) if isinstance(v, list | np.ndarray) else v
        )
    return s


def _describe(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """Report why two eval frames differ, most useful signal first."""
    if a.shape != b.shape:
        return [f"shape {a.shape} vs {b.shape}"]
    if list(a.columns) != list(b.columns):
        return [f"columns {list(a.columns)} vs {list(b.columns)}"]
    diffs = []
    # Column order is already known equal, so pair them off positionally --
    # `.items()` also gives a properly typed Series, which `a[col]` does not.
    for (col, left), (_, right) in zip(a.items(), b.items(), strict=True):
        x, y = _comparable(left), _comparable(right)
        # NaN != NaN, so compare null masks separately from values.
        ne = (x != y) & ~(x.isna() & y.isna())
        if n := int(ne.sum()):
            diffs.append(f"{col}: {n}/{len(a)} rows differ")
    return diffs


def main(baseline: Path, current: Path) -> int:
    base, curr = _relative_parquets(baseline), _relative_parquets(current)
    if not base:
        print(f"no eval_data.parquet found under {baseline}")
        return 1

    only_base, only_curr = (
        sorted(set(base) - set(curr)),
        sorted(set(curr) - set(base)),
    )
    shared = sorted(set(base) & set(curr))

    failed = []
    for rel in shared:
        diffs = _describe(
            pd.read_parquet(base[rel]), pd.read_parquet(curr[rel])
        )
        if diffs:
            failed.append((rel, diffs))

    for rel, diffs in failed:
        print(f"DIFF {rel.parent}")
        for d in diffs:
            print(f"       {d}")
    for rel in only_base:
        print(f"MISSING (in baseline only) {rel.parent}")
    for rel in only_curr:
        print(f"NEW     (in current only)  {rel.parent}")

    print(
        f"\n{len(shared) - len(failed)}/{len(shared)} shared files identical"
        f"{f', {len(failed)} differ' if failed else ''}"
        f"{f', {len(only_base)} missing' if only_base else ''}"
        f"{f', {len(only_curr)} new' if only_curr else ''}"
    )
    return 1 if (failed or only_base) else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
