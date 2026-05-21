"""Upload trained artifact bundles to HuggingFace Hub."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from huggingface_hub import HfApi

_OUTPUT_ROOT = Path("extra/output")
_DEFAULT_BUNDLE_TYPES = (
    "trained_methods",
    "pretrained_models",
    "trained_classifiers",
)


def _iter_bundle_paths(
    output_root: Path,
    bundle_types: list[str],
    initializer_tag: str | None,
    method: str | None,
    dataset: str | None,
) -> Iterator[Path]:
    for bundle_type in bundle_types:
        search_root = output_root / bundle_type
        if not search_root.exists():
            continue
        for bundle_path in sorted(search_root.rglob("*.bundle")):
            if not bundle_path.is_dir():
                continue
            path_str = str(bundle_path)
            if initializer_tag and f"/{initializer_tag}/" not in path_str:
                continue
            if method and f"/{method}/" not in path_str:
                continue
            if dataset and f"dataset-{dataset}" not in path_str:
                continue
            yield bundle_path


def _remote_path(bundle_path: Path, output_root: Path) -> str:
    return str(bundle_path.relative_to(output_root)) + ".zip"


def _zip_bundle(bundle_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(bundle_path.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(bundle_path.parent))


def _upload_bundle(
    api: HfApi,
    bundle_path: Path,
    output_root: Path,
    repo_id: str,
    existing: set[str],
    force: bool,
    dry_run: bool,
) -> None:
    remote_path = _remote_path(bundle_path, output_root)
    if remote_path in existing and not force:
        print(f"  skip  {remote_path}")
        return
    print(f"  upload {remote_path}")
    if dry_run:
        return
    with TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "bundle.zip"
        _zip_bundle(bundle_path, zip_path)
        api.upload_file(
            path_or_fileobj=str(zip_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
        )


def upload_bundles(
    output_root: Path,
    repo_id: str,
    bundle_types: list[str],
    initializer_tag: str | None,
    method: str | None,
    dataset: str | None,
    dry_run: bool,
    force: bool,
) -> None:
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    existing: set[str] = (
        set()
        if force
        else set(api.list_repo_files(repo_id, repo_type="model"))
    )
    for bundle_path in _iter_bundle_paths(
        output_root, bundle_types, initializer_tag, method, dataset
    ):
        _upload_bundle(
            api, bundle_path, output_root, repo_id, existing, force, dry_run
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload artifact bundles to HuggingFace Hub."
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-upload even if the file already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    upload_bundles(
        output_root=args.output_root,
        repo_id=args.repo_id,
        bundle_types=args.bundle_types,
        initializer_tag=args.initializer_tag,
        method=args.method,
        dataset=args.dataset,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    main()
