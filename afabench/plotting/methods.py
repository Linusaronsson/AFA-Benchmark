"""
Shared visual identity for the paper's figures.

Every method-shaped, dataset-shaped and mechanism-shaped constant lives here, so
figures cannot disagree about what a colour or a name means. The Hydra plotting
config mirrors the colours through ``method_color_overrides``;
``test_plotting_methods.py`` keeps the two equal and enforces the separation
floors below.

The palette is validated over *every* pair, not adjacent ones only, for
deuteranopia, protanopia and tritanopia: floors of 8 under CVD and 15 for normal
vision, in OKLab dE x100. Re-stepping one entry breaks that, so change the set
as a set.
"""

import matplotlib as mpl

METHOD_COLORS = {
    "aaco": "#D55E00",
    "aaco_doubly_robust": "#AA3377",
    "dime": "#0072B2",
    "dime_feature_marginal_ipw": "#56B4E9",
    "ol_with_mask": "#009E73",
    "ol_full_state": "#E69F00",
}

# ICLR 2026 is single column at 5.5in (iclr2026_conference.sty:50), so
# authoring here means \includegraphics[width=\textwidth] scales by 1.0 and a
# point size means what it says.
TEXT_WIDTH_IN = 5.5

# Height reserved below a faceted figure for its shared x label and legend.
# In inches rather than a fraction, so it does not grow with the row count.
LEGEND_STRIP_IN = 0.95

# Ink.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"
WEDGE = "#f0efec"
SURFACE = "#ffffff"


def apply_paper_style() -> None:
    """
    Set the rcParams every paper figure shares.

    ``fonttype 42`` matters beyond consistency: matplotlib defaults to Type 3,
    which arXiv flags and several venues reject, and Type 3 text neither copies
    nor searches.
    """
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.linewidth": 0.6,
            "text.color": INK,
            "axes.labelcolor": INK_MUTED,
            "axes.edgecolor": GRID,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


# The two OL states in the paper's own notation: main.tex:162 writes
# Q_k(x_S, S, a) and :228 writes Q_1^train(x_S, S, m, a).
METHOD_LABELS = {
    "aaco": "AACO",
    "aaco_doubly_robust": "AACO (doubly robust)",
    "dime": "DIME",
    "dime_feature_marginal_ipw": "DIME (feature-marginal IPW)",
    "ol_with_mask": "OL, $Q(x_S,S)$",
    "ol_full_state": "OL, $Q(x_S,S,m)$",
}

# Two registers, because a wide facet title and a narrow table column want
# different lengths. A dataset missing here shows its raw key, so add both.
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

# One hue light to dark, because the mechanisms are ordered rather than
# categorical: identification degrades along this list. Purple is clear of every
# method hue, so a mechanism cannot be misread as a method. Adjacent steps
# separate by at least 13.9 in OKLab dE x100 under normal vision and all three
# CVD simulations.
MECHANISM_COLORS = {
    "mcar": "#b7a3d4",
    "mar": "#8f6fba",
    "mnar_logistic": "#644191",
    "mnar_self": "#37225c",
}
