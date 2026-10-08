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

# Fast tests, without the workflow tier
test:
    uv run pytest .

# Fast tests, including the Snakemake workflow tier
test-with-workflow:
    uv run pytest . -m "not optional and not pipeline"

# Run all tests including expensive optional tests
test-full:
    uv run pytest . -m "optional or not optional"

# Commit gate: static checks + fast tests, plus the workflow tier when
# uncommitted changes touch what it tests
qa: (_qa "auto")

# Pre-PR gate: static checks + fast tests including the workflow tier
qa-full: (_qa "full")

# Plan the jobs of a pipeline invocation as a per-job CSV; takes the
# Snakemake arguments of the real run, plus --output <csv>
[positional-arguments]
estimate-compute *args:
    uv run python scripts/compute_estimate/estimate_compute.py "$@"

# Paths the workflow tier depends on
workflow_paths := "extra/workflow/ extra/conf/ test/workflow/ afabench/release/ afabench/fit/contract afabench/core/bundle_system/ afabench/core/output_layout afabench/core/workflow_settings afabench/compute_estimate/ scripts/compute_estimate/ afabench/core/job_record.py afabench/core/code_identity.py"

# Type check and tests run concurrently after the file-rewriting steps
_qa tier: fix
    #!/usr/bin/env bash
    set -uo pipefail
    tests="just test"
    if [ "{{ tier }}" = full ]; then
        tests="just test-with-workflow"
    else
        changed=$( (git diff --name-only HEAD; git ls-files --others --exclude-standard) | sort -u)
        for path in {{ workflow_paths }}; do
            if grep -q "^$path" <<< "$changed"; then
                echo "Changes under $path: including the workflow tier"
                tests="just test-with-workflow"
                break
            fi
        done
    fi
    logs=$(mktemp -d)
    trap 'kill $(jobs -p) 2>/dev/null; rm -rf "$logs"' EXIT
    trap 'exit 130' INT TERM
    just types > "$logs/types" 2>&1 &
    types=$!
    $tests > "$logs/tests" 2>&1 &
    tests_pid=$!
    wait "$types"; types_status=$?
    wait "$tests_pid"; tests_status=$?
    echo "=== just types (exit $types_status) ==="
    cat "$logs/types"
    echo "=== $tests (exit $tests_status) ==="
    cat "$logs/tests"
    exit $(( types_status || tests_status ))
