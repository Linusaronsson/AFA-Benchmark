# NHANES Mortality provenance

These files are an exact snapshot of the public CoAI NHANES data at commit
`4b5abba83d855d076ac9dc31e7b53d69c4429573`:

<https://github.com/suinleelab/coai/tree/4b5abba83d855d076ac9dc31e7b53d69c4429573/data/nhanes>

The snapshot contains 13,442 participants, 118 processed features, and a
binary ten-year mortality label. `source/COAI_LICENSE` is the upstream MIT
license. `source/feature_groups.txt` defines group identifiers 0 through 26;
the executable source therefore has 27 acquisition groups.

`schema.csv` is generated deterministically by:

```bash
uv run python scripts/dataset_generation/build_nhanes_schema.py
```

It resolves the unknown SGOT cost to the mean of the known source costs, adds
the upstream `0.001` low-cost floor once per group, and apportions that group
cost over its processed columns. The grouped unmasker sums those apportioned
costs, so acquiring a multi-column test pays exactly once.

Raw source missing values are imputed and normalized using training-only
statistics. The supplied missing/test-status indicator columns remain in the
same acquisition group as their measurement. Imposed missingness therefore
hides the processed source representation; it is not recovery of unrecorded
clinical values.
