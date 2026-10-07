# Format, lint, and type-check
check: fix types

# Format, lint and sync the type checker's excludes, rewriting files
fix:
    uv run ruff format .
    uv run ruff check . --fix
    uv run python scripts/dev/sync_excludes.py

# Type-check
types:
    uv run basedpyright --warnings

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
    trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$logs"' EXIT
    trap 'exit 130' INT TERM
    just types > "$logs/types" 2>&1 &
    types=$!
    just test > "$logs/tests" 2>&1 &
    tests=$!
    wait "$types"; types_status=$?
    wait "$tests"; tests_status=$?
    echo "=== just types (exit $types_status) ==="
    cat "$logs/types"
    echo "=== just test (exit $tests_status) ==="
    cat "$logs/tests"
    exit $(( types_status || tests_status ))
