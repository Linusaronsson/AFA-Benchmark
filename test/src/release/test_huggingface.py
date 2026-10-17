"""Host file sizes are available without fetching payload bytes."""

from collections.abc import Iterable

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
