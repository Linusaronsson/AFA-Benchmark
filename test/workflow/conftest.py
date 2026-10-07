"""Mark every test in this directory as part of the workflow tier."""

from pathlib import Path

import pytest

WORKFLOW_TESTS = Path(__file__).parent


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # tryfirst: the marker must exist before -m deselects items.
    for item in items:
        if item.path.is_relative_to(WORKFLOW_TESTS):
            item.add_marker(pytest.mark.workflow)
