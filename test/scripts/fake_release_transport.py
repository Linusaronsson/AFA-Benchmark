"""
An in-memory stand-in for the Hugging Face repository holding releases.

Like the hub, it stores file bytes only: file and directory mtimes are not
kept, and downloaded files are written fresh. `before_download` is called
with each file or folder path about to be downloaded, so a test can change
the host in the middle of a download.
"""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath


class FakeReleaseTransport:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commit_messages: list[str] = []
        self.before_download: Callable[[str], None] | None = None

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

    def list_folders(self, folder: str) -> list[str]:
        # A folder exists only as the parent of some file below it.
        return sorted(
            {
                PurePosixPath(path).relative_to(folder).parts[0]
                for path in self.files
                if PurePosixPath(path).parent.is_relative_to(folder)
                and PurePosixPath(path).parent != PurePosixPath(folder)
            }
        )

    def download_files(self, paths: Sequence[str], local_dir: Path) -> None:
        for path in paths:
            self._notify(path)
            if path not in self.files:
                msg = f"No file {path!r} on the fake host."
                raise FileNotFoundError(msg)
            self._write(path, local_dir)

    def download_folder(self, folder: str, local_dir: Path) -> None:
        self._notify(folder)
        for path in list(self.files):
            if path.startswith(f"{folder}/"):
                self._write(path, local_dir)

    def folder_url(self, folder: str) -> str:
        return f"https://hub.invalid/{folder}"

    def _notify(self, path: str) -> None:
        if self.before_download is not None:
            self.before_download(path)

    def _write(self, path: str, local_dir: Path) -> None:
        target = local_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[path])
