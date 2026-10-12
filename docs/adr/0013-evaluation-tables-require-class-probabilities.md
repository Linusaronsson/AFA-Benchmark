---
status: accepted
---

# Evaluation tables require class probabilities

Evaluation retains class-index-ordered probability vectors for every recorded
time step, alongside derived predicted classes. Classifiers declare whether
they produce logits or probabilities; upstream adapters convert declared logits,
and the shared evaluator validates probabilities without guessing semantics.
This supports score-based metrics such as AUROC without reevaluating methods
whenever a new metric is added.

This **amends ADR 0002's compatibility promise for evaluation tables**:
class-only raw and transformed tables are no longer supported. Their class
indices cannot reconstruct the ranking scores AUROC needs. We chose a single
probability-bearing contract over a legacy accuracy/F1-only reader to avoid
maintaining two evaluation-table formats or presenting fabricated probabilities.
Previously trained bundles need not be retrained for this schema change;
[reevaluate class-only tables](../how-to/run_the_pipeline.md#reevaluate-class-only-tables)
and regenerate downstream tables and figures. ADR 0002's artifact provenance
and legacy-bundle compatibility decisions are unchanged.
