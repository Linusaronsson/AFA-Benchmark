# Contributing

This document describes what's expected of anyone developing in this
repository, human or AI agent.

## Setup

```bash
# Install dependencies
uv sync
```

## Before opening a PR

Run the full quality gate and make sure it passes:

```bash
just qa-full
```

This runs formatting, linting, type checking, and tests together, including
the Snakemake workflow tests. Before each commit, `just qa` is enough: it runs
the workflow tests only when your changes touch the workflow. See
`AGENTS.md` for the individual commands (`ruff format`, `ruff check`,
`basedpyright`, `pytest`) if you need to debug a specific failure.

## Engineering skills

Development in this repo follows the `matt-pocock` skill set (e.g. `tdd`,
`code-review`, `diagnosing-bugs`, `domain-modeling`, `codebase-design`,
`resolving-merge-conflicts`, `writing-for-agents`). Contributors, human or
agent, are expected to use these skills where applicable rather than ad hoc
approaches.

If you don't have them installed yet, install them from
<https://github.com/mattpocock/skills> following that repo's instructions.

AI agents working in this repo should also read `AGENTS.md`, which covers
project conventions, the required `just qa` quality gate, and pointers to the
issue tracker, triage labels, and domain docs under `docs/agents/`.
