"""
The Hugging Face dataset repository that hosts benchmark releases.

This is the only module that talks to Hugging Face. Credentials come from
`huggingface_hub`'s usual sources (`HF_TOKEN` or `hf auth login`) and are
needed to publish only: public releases download anonymously.
"""

import hashlib
from collections.abc import Mapping, Sequence
from itertools import batched
from pathlib import Path

import httpx
from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    constants,
    hf_hub_download,
    snapshot_download,
)
from huggingface_hub.hf_api import RepoFile

REPO_TYPE = "dataset"


class HuggingFaceTransport:
    def __init__(self, repo_id: str) -> None:
        self.repo_id: str = repo_id
        self._api: HfApi = HfApi()

    def file_exists(self, path: str) -> bool:
        return self._api.file_exists(self.repo_id, path, repo_type=REPO_TYPE)

    def upload_files(
        self, files: Mapping[str, Path | bytes], message: str
    ) -> None:
        try:
            self._api.create_commit(
                self.repo_id,
                [
                    CommitOperationAdd(
                        path_in_repo=path, path_or_fileobj=content
                    )
                    for path, content in files.items()
                ],
                commit_message=message,
                repo_type=REPO_TYPE,
            )
        except httpx.ReadTimeout:
            # A lost commit response does not mean the commit failed. Verify
            # content, not just the manifest's existence, before accepting it.
            try:
                committed = self._files_match(files)
            except (httpx.HTTPError, OSError):
                # Keep the original timeout if verification is unavailable.
                committed = False
            if not committed:
                raise

    def _files_match(self, files: Mapping[str, Path | bytes]) -> bool:
        revision = self._api.repo_info(self.repo_id, repo_type=REPO_TYPE).sha
        if revision is None or not files:
            return False
        # Pin all batches to one commit so concurrent publication cannot
        # produce a verification assembled from different repository states.
        for paths in batched(files, 100):
            entries = self._api.get_paths_info(
                self.repo_id,
                list(paths),
                repo_type=REPO_TYPE,
                revision=revision,
            )
            remote = {
                entry.path: entry
                for entry in entries
                if isinstance(entry, RepoFile)
            }
            if set(remote) != set(paths):
                return False
            for path in paths:
                if not _content_matches(files[path], remote[path]):
                    return False
        return True

    def file_sizes(self, folder: str) -> dict[str, int]:
        return {
            entry.path: entry.size
            for entry in self._api.list_repo_tree(
                self.repo_id,
                path_in_repo=folder,
                recursive=True,
                repo_type=REPO_TYPE,
            )
            if isinstance(entry, RepoFile)
        }

    def download_files(self, paths: Sequence[str], local_dir: Path) -> None:
        for path in paths:
            hf_hub_download(
                self.repo_id,
                path,
                repo_type=REPO_TYPE,
                local_dir=local_dir,
            )

    def download_folder(self, folder: str, local_dir: Path) -> None:
        snapshot_download(
            self.repo_id,
            repo_type=REPO_TYPE,
            allow_patterns=f"{folder}/*",
            local_dir=local_dir,
        )

    def folder_url(self, folder: str) -> str:
        return (
            f"{constants.ENDPOINT}/datasets/{self.repo_id}/tree/main/{folder}"
        )


def _content_matches(content: Path | bytes, remote: RepoFile) -> bool:
    size = (
        content.stat().st_size if isinstance(content, Path) else len(content)
    )
    if size != remote.size:
        return False
    if remote.lfs is not None:
        digest = hashlib.sha256()
        expected = remote.lfs.sha256
    else:
        digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
        expected = remote.blob_id
    if isinstance(content, Path):
        with content.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        digest.update(content)
    return digest.hexdigest() == expected
