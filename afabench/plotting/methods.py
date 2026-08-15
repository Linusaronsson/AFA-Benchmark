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

# ICLR 2026 is single column at 5.5in (iclr2026_conference.sty:50), and
# \includegraphics[width=\textwidth] scales a figure to it. Author at this width
# so nothing is resized on inclusion and a point size means what it says. The
# 7.16in the older scripts used is an IEEEtran double-column leftover.
TEXT_WIDTH_IN = 5.5

# Height reserved below a faceted figure for its shared x label and legend.
# In inches, not a fraction of the figure: as a fraction it was tuned at one row
# of facets and wasted an inch of white once the dataset count forced three.
LEGEND_STRIP_IN = 0.95

# Name the two OL states in the paper's own notation, main.tex:162 writes
# Q_k(x_S, S, a) and :228 writes Q_1^train(x_S, S, m, a). "OL(with-mask)" did
# not say what differed, and the difference is the whole point of the contrast.
METHOD_LABELS = {
    "aaco": "AACO",
    "aaco_doubly_robust": "AACO (doubly robust)",
    "dime": "DIME",
    "dime_feature_marginal_ipw": "DIME (feature-marginal IPW)",
    "ol_with_mask": "OL, $Q(x_S,S)$",
    "ol_full_state": "OL, $Q(x_S,S,m)$",
}

# Dataset names in two registers, because a wide facet title and a narrow table
# column want different lengths. Five scripts carried private copies of these,
# which is why adding MiniBooNE printed a correct title in one figure and the
# raw key in four others. A dataset missing here shows its key, so add both.
DATASET_LABELS = {
    "cube": "CUBE",
    "cube_nm": "CUBE-NM",
    "cube_nonuniform_costs": "CUBE non-uniform cost",
    "heart_disease": "Heart disease",
    "actg": "ACTG175",
    "diabetes": "Diabetes",
    "nhanes_mortality": "NHANES mortality",
    "miniboone": "MiniBooNE",
    "ckd": "CKD",
    "physionet": "PhysioNet",
}

# Compact forms for narrow panels and for table columns, so they stay LaTeX-safe.
DATASET_LABELS_SHORT = {
    **DATASET_LABELS,
    "cube_nonuniform_costs": "CUBE-NUC",
    "nhanes_mortality": "NHANES",
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
