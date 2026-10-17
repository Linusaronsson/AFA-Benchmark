---
status: accepted
---

# A release carries an explicitly runnable benchmark configuration

A release includes a runnable configuration with pipeline-level defaults frozen
when jobs execute, separate from its provenance record. Downloads rebase its
output root to their destination; the launcher applies it only when explicitly
selected with `--configfile`. Publisher-specific execution, site, logging and
stage switches are not replayed. This avoids reconstructing profiles without
making a downloaded cache silently choose what an unrelated run computes.
The configuration does not package code, environments or transitive Hydra
configs, and is not a substitute for per-artifact provenance.

The launcher owns the dataset preparation/reuse boundary: one user invocation
may run dataset generation first, then construct a fresh DAG with generation
disabled and content-checked release reuse enabled. Validation must finish before
scheduling consumers. This retains the safety boundary of ADR 0012 without
requiring users to coordinate stage switches across commands. Dry runs and
estimates can show reuse conditional on generation producing matching bytes;
they cannot validate bytes that do not yet exist. Execution cannot enable that
preview exemption.

This amends ADR 0012's prohibition on applying release configuration and its
requirement for manually separate invocations. The release remains a cache:
missing work is computed normally, artifact provenance determines coverage, and
a mismatched dataset fails rather than silently combining incompatible outputs.
