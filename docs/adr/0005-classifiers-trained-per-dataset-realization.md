---
status: accepted
---

# Classifiers are trained per dataset realization

Since commit `7b4c2e74` ("build: only train classifier once per dataset"),
the external classifier of a dataset was trained once, on the train and val
splits of dataset realization 0, and handed to pretraining, training and
evaluation on every dataset realization. Method-specific classifiers did the
same. While the pipeline only ran synthetic datasets, each realization was a
fresh sample and the shortcut was harmless. A real-world dataset's
realizations are reshuffles of one pool, so roughly 60% of the test split of
realization k ≥ 1 was in the classifier's training split, and every
external-classifier result on those realizations was evaluated partly on
instances the classifier had seen (#73).

We decided that every dataset realization gets its own external classifier,
and its own method-specific classifiers, trained on that realization's train
and val splits only, with the realization index as seed. Every pipeline stage
working on realization k receives the classifiers of realization k. The rule
holds for every dataset, real-world or synthetic: synthetic datasets did not
leak, but one rule needs no remembering, and classifier variability becomes
part of the realization-to-realization variance like every other seed.

Classifier bundles are named like the other outputs of a realization:
`dataset-<key>+realization_index-<k>.bundle` and
`method-<method>+dataset-<key>+realization_index-<k>.bundle`. This amends
ADR 0002, whose table of observed behaviours records the external
classifier's seed 0 as policy and whose out-of-scope note records the leak:
the seed is now the dataset realization index.

## Consequences

- Classifier training compute grows linearly with the number of dataset
  realizations (5 in the shipped configs). It is small next to method
  training.
- Rerunning a published config changes the external-classifier results of
  real-world datasets on realizations 1 and above.

## Considered options

- **One classifier per dataset, trained on realization 0.** Rejected: leaks
  the test split of every later realization of a real-world dataset into the
  classifier's training data.
- **Per-realization classifiers only for real-world datasets.** Rejected: two
  rules for the same artifact, and a dataset changing kind would silently
  change which one applies.
- **Hold the test split fixed across realizations.** Out of scope: it
  changes what a dataset realization is, for every stage, not just the
  classifier.
