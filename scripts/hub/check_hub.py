"""Compare HuggingFace Hub bundle contents against local output."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from huggingface_hub import HfApi

from scripts.hub.upload_bundles import (
    _DEFAULT_BUNDLE_TYPES,
    _OUTPUT_ROOT,
    _iter_bundle_paths,
    _remote_path,
)


def _hub_bundle_paths(
    api: HfApi,
    repo_id: str,
    bundle_types: list[str],
    initializer_tag: str | None,
    method: str | None,
    dataset: str | None,
) -> set[str]:
    all_files = set(api.list_repo_files(repo_id, repo_type="model"))
    prefix = tuple(f"{t}/" for t in bundle_types)
    candidates: Iterator[str] = (
        f
        for f in all_files
        if f.endswith(".bundle.zip") and f.startswith(prefix)
    )
    if initializer_tag:
        candidates = (f for f in candidates if f"/{initializer_tag}/" in f)
    if method:
        candidates = (f for f in candidates if f"/{method}/" in f)
    if dataset:
        candidates = (f for f in candidates if f"dataset-{dataset}" in f)
    return set(candidates)


def check_hub(
    output_root: Path,
    repo_id: str,
    bundle_types: list[str],
    initializer_tag: str | None,
    method: str | None,
    dataset: str | None,
) -> bool:
    """
    Compare HF Hub bundles against local output.

    Reports local bundles not yet uploaded and HF Hub bundles missing locally.
    Returns True only if all local bundles are present on HF Hub.
    """
    api = HfApi()
    local = {
        _remote_path(p, output_root)
        for p in _iter_bundle_paths(
            output_root, bundle_types, initializer_tag, method, dataset
        )
    }
    remote = _hub_bundle_paths(
        api, repo_id, bundle_types, initializer_tag, method, dataset
    )

    not_uploaded = sorted(local - remote)
    missing_locally = sorted(remote - local)

    if not_uploaded:
        print(
            f"{len(not_uploaded)} local bundle(s) not yet uploaded to {repo_id}:"
        )
        for path in not_uploaded:
            print(f"  {path}")
    else:
        print(f"All {len(local)} local bundle(s) are present on {repo_id}.")

    if missing_locally:
        print(
            f"\n{len(missing_locally)} bundle(s) on {repo_id} missing locally:"
        )
        for path in missing_locally:
            print(f"  {path}")

    return not not_uploaded


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HF Hub bundle contents against local output."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF Hub repo ID, e.g. myorg/afa-benchmark-checkpoints",
    )
    parser.add_argument("--output-root", type=Path, default=_OUTPUT_ROOT)
    parser.add_argument(
        "--bundle-types", nargs="+", default=list(_DEFAULT_BUNDLE_TYPES)
    )
    parser.add_argument(
        "--initializer-tag",
        help="Filter by initializer tag, e.g. initializer-cold",
    )
    parser.add_argument("--method", help="Filter by method name")
    parser.add_argument("--dataset", help="Filter by dataset name")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ok = check_hub(
        output_root=args.output_root,
        repo_id=args.repo_id,
        bundle_types=args.bundle_types,
        initializer_tag=args.initializer_tag,
        method=args.method,
        dataset=args.dataset,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
