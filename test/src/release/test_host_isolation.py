"""Only the release tooling talks to the release host (#36, #38)."""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).parents[3]
HOST_MODULES = ("huggingface_hub", "afabench.release.huggingface")
# The adapter itself and the release command that selects it.
ALLOWED = {
    Path("afabench/release/huggingface.py"),
    Path("scripts/release/snapshot.py"),
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_no_pipeline_code_imports_the_release_host() -> None:
    offenders = [
        path.relative_to(REPO_ROOT)
        for directory in ["afabench", "scripts", "workflow"]
        for path in sorted((REPO_ROOT / directory).rglob("*.py"))
        if path.relative_to(REPO_ROOT) not in ALLOWED
        and any(
            module == host or module.startswith(f"{host}.")
            for module in imported_modules(path)
            for host in HOST_MODULES
        )
    ]

    assert offenders == []
