# Format, lint, and type-check
check: fix
    uv run basedpyright --warnings

# Format and lint, rewriting files
fix:
    uv run ruff format .
    uv run ruff check . --fix
    uv run python scripts/dev/sync_excludes.py

# Fast tests
test:
    uv run pytest .

# Run all tests including expensive optional tests
test-full:
    uv run pytest . -m "optional or not optional"

# QA = static checks + fast tests; type check and tests run concurrently
qa: fix
    #!/usr/bin/env bash
    set -uo pipefail
    logs=$(mktemp -d)
    trap 'rm -rf "$logs"' EXIT
    uv run basedpyright --warnings > "$logs/types" 2>&1 &
    types=$!
    uv run pytest . > "$logs/tests" 2>&1 &
    tests=$!
    wait "$types"; types_status=$?
    wait "$tests"; tests_status=$?
    cat "$logs/types" "$logs/tests"
    echo "basedpyright exit $types_status, pytest exit $tests_status"
    exit $(( types_status || tests_status ))
