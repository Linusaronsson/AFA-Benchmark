# Terminology

Descriptions are given in the context of patch-based image unmasking. A 28x28 image reduced to 7x7 patches, each patch having size 4x4.

---

Selection: which patch to unmask next, 0 for the first patch, 1 for the second, etc.

Action: 0 to stop acquiring new features, or i>0 which patch to unmask next. Hence, n_actions = n_selections + 1.

Unmasker: takes a *selection* and the current feature mask and returns a new feature mask. For example, a patch-based image unmasker might receive the number 3 to unmask the **fourth** patch with some probability.

Initializer: decides the initial feature mask. While starting with no features is common, it could be interesting to investigate other scenarios.
