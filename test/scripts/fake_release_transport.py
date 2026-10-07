"""
An in-memory stand-in for the Hugging Face repository holding releases.

Like the hub, it stores file bytes only: file and directory mtimes are not
kept, and downloaded files are written fresh.
"""

from collections.abc import Mapping
from pathlib import Path


class FakeReleaseTransport:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commit_messages: list[str] = []

    def file_exists(self, path: str) -> bool:
        return path in self.files

    def upload_files(
        self, files: Mapping[str, Path | bytes], message: str
    ) -> None:
        for path, content in files.items():
            self.files[path] = (
                content if isinstance(content, bytes) else content.read_bytes()
            )
        self.commit_messages.append(message)

    def download_folder(self, folder: str, local_dir: Path) -> None:
        for path, content in self.files.items():
            if path.startswith(f"{folder}/"):
                target = local_dir / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

    def folder_url(self, folder: str) -> str:
        return f"https://hub.invalid/{folder}"
