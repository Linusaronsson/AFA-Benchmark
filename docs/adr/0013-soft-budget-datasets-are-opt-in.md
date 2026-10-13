---
status: accepted
---

# A method runs the soft-budget setting only on the datasets it lists

Each method's `soft_budget_params` profile entry has a `default` that every
dataset not named in it receives, and methods left datasets out through
`soft_budget_ignored_datasets`. A newly added dataset therefore ran in the
soft-budget setting for every method with non-empty defaults (AACO, the
dummy methods and, in the `fast` and `single` profiles, JAFA and ODIN) at
parameter values tuned on other datasets, which gives meaningless points
rather than no points. Adding METABRIC (#132), whose soft-budget parameters
nobody has tuned, made this concrete.

We decided that soft budgets are opt-in per method and dataset: the method
option `soft_budget_datasets` lists the datasets on which the method runs
the soft-budget setting, and a dataset it does not list gets no soft-budget
runs, whatever the soft-budget profiles give it. The `default` fallback in
the profiles stays, so a listed dataset still needs no entry of its own.
The hard-budget setting stays opt-out (`hard_budget_ignored_datasets`): hard
budgets need no tuning, so running a new dataset there by default is useful.
The lists were filled so that every existing profile resolves to exactly
the runs it had before.

## Considered options

- **Remove `default` from the soft-budget profiles**, so only datasets named
  there run. Rejected: every profile, including the short `fast` and
  `single` ones, would have to spell out every dataset for every method, and
  which datasets run would be spread over five files instead of one.
- **Keep opt-out and add each new dataset to the ignore lists.** Rejected:
  forgetting to do so is silent, which is the failure this decision removes.

## Consequences

- Enabling the soft-budget setting for a method on a dataset takes two
  steps: list the dataset in the method's `soft_budget_datasets`, and give
  it tuned parameters in the soft-budget profiles if the `default` does not
  fit.
- The two settings are now asymmetric: hard-budget lists name exclusions,
  the soft-budget list names inclusions.
