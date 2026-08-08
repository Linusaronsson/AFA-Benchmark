"""
Shared visual identity for the missing-data methods.

Four plotting scripts used to carry four private palettes for the same methods,
which is how ol_with_mask and ol_full_state ended up as two near-identical
oranges in one figure and as one colour in another. Everything method-shaped
lives here instead, and the Hydra plotting config mirrors the colours through
``method_color_overrides`` (``test_plotting_methods.py`` keeps the two equal).

The palette is validated over *every* pair rather than adjacent ones only, for
deuteranopia, protanopia and tritanopia: worst pair 10.3 (deutan) and 15.6
(normal vision) in OKLab dE x100, against floors of 8 and 15. Re-stepping a
single entry breaks that guarantee, so change the set as a set.
"""

METHOD_COLORS = {
    "aaco": "#D55E00",
    "aaco_doubly_robust": "#AA3377",
    "dime": "#0072B2",
    "dime_feature_marginal_ipw": "#56B4E9",
    "ol_with_mask": "#009E73",
    "ol_full_state": "#E69F00",
}

# Name the state contents. "OL(with-mask)" and "OL(full-state)" do not tell a
# reader what differs, and the difference is the whole point of the contrast.
METHOD_LABELS = {
    "aaco": "AACO",
    "aaco_doubly_robust": "AACO (doubly robust)",
    "dime": "DIME",
    "dime_feature_marginal_ipw": "DIME (feature-marginal IPW)",
    "ol_with_mask": "OL (values + acquired mask)",
    "ol_full_state": "OL (+ legal-action mask)",
}

# Redundant channels, so identity survives greyscale printing and CVD.
METHOD_MARKERS = {
    "aaco": "o",
    "aaco_doubly_robust": "D",
    "dime": "^",
    "dime_feature_marginal_ipw": "s",
    "ol_with_mask": "v",
    "ol_full_state": "P",
}

# Solid is a method, dashed is a reweighting control of a method.
METHOD_LINESTYLES = {
    "aaco": "solid",
    "aaco_doubly_robust": "dashed",
    "dime": "solid",
    "dime_feature_marginal_ipw": "dashed",
    "ol_with_mask": "solid",
    "ol_full_state": "solid",
}

# The four primary methods every induced-missingness cell runs. The two
# reweighting controls are trained on the restricted view only, so figures that
# compare training views iterate over these.
PRIMARY_METHODS = ("aaco", "dime", "ol_with_mask", "ol_full_state")

# Induced mechanisms, ordered by how much of the damage restoration recovers,
# which is also the order in which identification degrades. Self-masking MNAR is
# last because an entry's own value decides whether it is observed, so the
# conditional the generator needs is not identified (prop:mnar).
INDUCED_MECHANISMS = ("mcar", "mar", "mnar_logistic", "mnar_self")
MECHANISM_LABELS = {
    "mcar": "MCAR",
    "mar": "MAR",
    "mnar_logistic": "MNAR (logistic)",
    "mnar_self": "MNAR (self-masking)",
    "native": "Native",
}
MECHANISM_MARKERS = {
    "mcar": "o",
    "mar": "^",
    "mnar_logistic": "s",
    "mnar_self": "D",
}
