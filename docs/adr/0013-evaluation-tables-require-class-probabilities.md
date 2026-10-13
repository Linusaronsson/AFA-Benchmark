---
status: accepted
---

# Evaluation tables require class probabilities

Evaluation retains class-index-ordered probability vectors for every recorded
time step and stores no class decisions; each consumer chooses its own
decision rule. Classifiers declare whether
they produce logits or probabilities; upstream adapters convert declared logits,
and the shared evaluator validates probabilities without guessing semantics.
Retaining full vectors costs more storage than predicted classes would,
but supports score-based metrics such as AUROC without reevaluating methods
whenever a new metric is added. Explicit output declarations avoid confusing
normalized probabilities with logits whose values happen to lie in [0, 1].
