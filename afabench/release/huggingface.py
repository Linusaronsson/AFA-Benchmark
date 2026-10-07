"""
The Hugging Face dataset repository that hosts benchmark releases.

This is the only module that talks to Hugging Face. Credentials come from
`huggingface_hub`'s usual sources (`HF_TOKEN` or `hf auth login`) and are
needed to publish only: public releases download anonymously.
"""

from collections.abc import Mapping
from pathlib import Path

from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    constants,
    snapshot_download,
)

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
        self._api.create_commit(
            self.repo_id,
            [
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=content)
                for path, content in files.items()
            ],
            commit_message=message,
            repo_type=REPO_TYPE,
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
