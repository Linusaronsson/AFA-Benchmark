# Evaluation input cloning: assessment (GH issue #35)

## Question

`process_batch` (`afabench/evaluation/eval.py`) clones its three batch-level
inputs on entry:

```python
features = features.clone()
feature_mask = initial_feature_mask.clone()
masked_features = initial_masked_features.clone()
```

This was flagged as a speculative optimization target: do these copies cause
meaningful avoidable overhead, and can any be removed without changing
evaluation behavior?

## Per-copy analysis

- **`feature_mask = initial_feature_mask.clone()`**: required. The loop
  mutates `feature_mask` in place via indexed assignment
  (`feature_mask[active_indices] = step.feature_mask`). Without the clone,
  this would mutate the caller's `initial_feature_mask` tensor in place.
- **`masked_features = initial_masked_features.clone()`**: required, for the
  same reason (`masked_features[active_indices] = step.masked_features`).
- **`features = features.clone()`**: *not* required for correctness.
  `features` is only read in `process_batch` and `single_afa_step` (via
  advanced indexing, which always copies in PyTorch); it is never written to
  in place. Removing this clone would not change behavior or risk aliasing
  the caller's tensor.

So one of the three copies (`features`) is provably unnecessary. The other
two are load-bearing and must stay regardless of performance.

## Measured overhead

`scripts/dev/benchmark_process_batch_cloning.py` measures the cost of all
three clones together against total `process_batch` time, across workload
sizes mirroring the pinned AACO evaluation batch sizes in
`test/src/common/test_workflow_settings.py` (32 for image datasets with 784
features, 128 for tabular datasets with 20 features), with the episode length
capped by `selection_budget` to a realistic handful of acquisitions.

Device: CPU (no CUDA device was available in the environment used to produce
these numbers; the loop is Python-level per-timestep work that does not
change character on GPU, so the *relative* overhead is expected to transfer).

| batch_size | n_features | selection_budget | total / call | three clones / call | % of total |
|---|---|---|---|---|---|
| 32  | 784 | 10 | 23.30 ms  | 0.0221 ms | 0.095% |
| 128 | 20  | 10 | 26.44 ms  | 0.0154 ms | 0.058% |
| 256 | 784 | 20 | 149.40 ms | 0.2555 ms | 0.171% |

Across all three workloads, all three clones combined account for well under
0.2% of total batch evaluation time. The per-timestep Python-level work in
the acquisition loop (calling `afa_action_fn`/`afa_unmask_fn`, indexed
assignments, active-set bookkeeping) dominates by a factor of roughly 500-1700x.

## Decision

**Retain all three clones, including the unnecessary `features.clone()`.**

Removing `features.clone()` would save roughly 0.03-0.08 ms per batch out of
measured totals of 23-150 ms — not a measurable or practically beneficial
improvement in any workload tested. Keeping it also protects `process_batch`
against regressions: it is a public function exercised directly by tests
(`test/src/common/eval/test_process_batch.py`) and future code changes
inside `single_afa_step` could start writing into `features` without anyone
noticing the broken caller-ownership contract.

This matches the issue's own criterion: a copy should only be removed if it
is both unnecessary *and* beneficial to remove. `features.clone()` is
unnecessary but not beneficial to remove, so it stays.

## Limitations

- No CUDA device was available to measure GPU overhead directly; the
  conclusion that clone cost is dominated by per-step Python overhead is
  expected to hold there too, but was not directly measured.
- The benchmark uses a synthetic dummy action/unmask function; real AFA
  methods (e.g. neural network forward passes) make the per-step cost even
  larger relative to the clones, reinforcing rather than undermining the
  conclusion.

## Regression coverage

`test/src/common/eval/test_process_batch_input_preservation.py` asserts that
`process_batch` leaves `features`, `initial_feature_mask`, and
`initial_masked_features` unchanged in both the hard-budget
(`selection_budget` set) and soft-budget (`selection_budget=None`) settings,
and that `eval_afa_method` leaves the underlying dataset's feature tensor
unchanged end-to-end.
