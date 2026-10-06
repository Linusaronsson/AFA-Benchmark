# Evaluation owns hard-budget forced acquisition

`eval_afa_method` always forces acquisition when a hard budget is supplied,
regardless of the method's capabilities or an explicit forcing flag. The shared
evaluation module replaces stop with the first available selection and asserts
that no stop remains before applying the budget cutoff. This prevents adapter
configuration from silently allowing a method to under-spend its hard budget.

Method-internal forcing is an optional optimisation: the script may enable it
for RL methods and AACO to retain their preferred selection rather than the
fallback selection. It is never trusted for correctness; even these methods
have their stop actions overridden. We reject separate, mutually exclusive
method-backed and fallback enforcement paths because their configuration was
the source of issue #42.

## Callable evaluation interface

Retain the callable protocols rather than require full method, Unmasker and
Initializer objects. Evaluation only needs their episode operations; requiring
objects also demands unrelated persistence, device and seed interfaces from
lightweight adapters and tests. Align initializer label nullability and its
default with `AFAInitializer.initialize`. The duplication is a deliberate cost
of keeping this narrow seam, not a second domain model.

## Consequences

Soft-budget evaluation continues to allow voluntary stop. Lower-level
`process_batch` and `single_afa_step` retain explicit forcing controls for
step/batch callers. A budget is a cap on accumulated selection cost, not a
promise to spend an unrepresentable amount: an action that would exceed it is
still overridden to stop and recorded as a forced stop. Existing repeated
selection semantics remain unchanged when every selection has been performed,
including for stochastic Unmaskers. No cost-aware alternative-selection search
is introduced.

The published/kdd26 results investigation in issue #42 is explicitly outside
this change at the operator's request; no claim is made
about whether those results are affected.
