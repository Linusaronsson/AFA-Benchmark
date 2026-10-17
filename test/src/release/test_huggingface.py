"""Hugging Face metadata supports size queries and commit verification."""

import hashlib
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub import HfApi
from huggingface_hub.hf_api import RepoFile, RepoFolder

from afabench.release.huggingface import HuggingFaceTransport


def test_file_sizes_lists_release_recursively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def list_repo_tree(
        _self: HfApi,
        repo_id: str,
        *,
        path_in_repo: str,
        recursive: bool,
        repo_type: str,
    ) -> Iterable[RepoFile | RepoFolder]:
        assert repo_id == "afabench/results"
        assert path_in_repo == "releases/example"
        assert recursive
        assert repo_type == "dataset"
        yield RepoFolder(path="releases/example/output", oid="tree")
        yield RepoFile(
            path="releases/example/output/plot.pdf", size=1234, oid="blob"
        )
        yield RepoFile(
            path="releases/example/output/empty", size=0, oid="empty"
        )

    monkeypatch.setattr(HfApi, "list_repo_tree", list_repo_tree)

    assert HuggingFaceTransport("afabench/results").file_sizes(
        "releases/example"
    ) == {
        "releases/example/output/plot.pdf": 1234,
        "releases/example/output/empty": 0,
    }


@pytest.mark.parametrize(
    "remote_state",
    [
        "matching",
        "missing",
        "partial",
        "changed",
        "changed-lfs",
        "wrong-size",
        "unavailable",
    ],
)
def test_commit_timeout_checks_remote_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    remote_state: str,
) -> None:
    payload = tmp_path / "model.bin"
    payload.write_bytes(b"model")
    files = {"release/manifest.json": b"{}", "release/model.bin": payload}
    timeout = httpx.ReadTimeout("The read operation timed out")
    commits = []

    def create_commit(_self: HfApi, *args: object, **kwargs: object) -> None:
        commits.append((args, kwargs))
        raise timeout

    def repo_info(
        _self: HfApi, repo_id: str, *, repo_type: str
    ) -> SimpleNamespace:
        assert repo_id == "afabench/results"
        assert repo_type == "dataset"
        if remote_state == "unavailable":
            message = "Verification timed out"
            raise httpx.ReadTimeout(message)
        return SimpleNamespace(sha="committed-revision")

    def get_paths_info(
        _self: HfApi,
        repo_id: str,
        paths: list[str],
        *,
        revision: str,
        repo_type: str,
    ) -> list[RepoFile]:
        assert repo_id == "afabench/results"
        assert revision == "committed-revision"
        assert repo_type == "dataset"
        assert set(paths) == set(files)
        if remote_state == "missing":
            return []
        manifest = b"[]" if remote_state == "changed" else b"{}"
        entries = [
            RepoFile(
                path="release/manifest.json",
                size=len(manifest),
                oid=hashlib.sha1(
                    b"blob 2\0" + manifest, usedforsecurity=False
                ).hexdigest(),
            ),
            RepoFile(
                path="release/model.bin",
                size=6 if remote_state == "wrong-size" else 5,
                oid="lfs-pointer",
                lfs={
                    "oid": hashlib.sha256(
                        b"other" if remote_state == "changed-lfs" else b"model"
                    ).hexdigest(),
                    "size": 5,
                    "pointerSize": 130,
                },
            ),
        ]
        return entries[:1] if remote_state == "partial" else entries

    monkeypatch.setattr(HfApi, "create_commit", create_commit)
    monkeypatch.setattr(HfApi, "repo_info", repo_info)
    monkeypatch.setattr(HfApi, "get_paths_info", get_paths_info)
    transport = HuggingFaceTransport("afabench/results")
    if remote_state == "matching":
        transport.upload_files(files, message="Publish release")
    else:
        with pytest.raises(httpx.ReadTimeout) as caught:
            transport.upload_files(files, message="Publish release")
        assert caught.value is timeout
    assert len(commits) == 1
