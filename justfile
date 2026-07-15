# Format, lint, and type-check
check:
    uv run ruff format .
    uv run ruff check . --fix
    uv run basedpyright --warnings

# Fast tests
test:
    uv run pytest .

# Run all tests including expensive optional tests
test-full:
    uv run pytest . -m "optional or not optional"

# QA = static checks + fast tests
qa:
    just check
    just test
