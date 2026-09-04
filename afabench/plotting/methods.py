"""
Shared visual identity for the paper's figures.

Every method-shaped, dataset-shaped and mechanism-shaped constant lives here, so
figures cannot disagree about what a colour or a name means. The Hydra plotting
config mirrors the colours through ``method_color_overrides``;
``test_plotting_methods.py`` keeps the two equal and enforces the floors below.

Hue is a *family*, not a method: eleven mutually separated hues cannot be calm,
and a ninth generated hue is indistinguishable from an existing one under CVD.
Methods inside a family share a hue and are told apart by marker and by row
position. The six family hues are validated over every pair, not adjacent ones
only, for deuteranopia, protanopia and tritanopia: floors of 8 under CVD and 15
for normal vision, in OKLab dE x100.
"""

import matplotlib as mpl

# Okabe-Ito, one hue per method family.
FAMILY_COLORS = {
    "aaco": "#D55E00",
    "dime": "#56B4E9",
    "gdfs": "#009E73",
    "jafa": "#882255",
    "ol": "#E69F00",
    "odin": "#7570B3",
}

METHOD_FAMILIES = {
    "aaco": "aaco",
    "aaco_doubly_robust": "aaco",
    "dime": "dime",
    "dime_feature_marginal_ipw": "dime",
    "gdfs": "gdfs",
    "jafa": "jafa",
    "jafa_full_state": "jafa",
    "ol_with_mask": "ol",
    "ol_full_state": "ol",
    "odin_model_free": "odin",
    "odin_model_free_full_state": "odin",
}

METHOD_COLORS = {
    method: FAMILY_COLORS[family] for method, family in METHOD_FAMILIES.items()
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


# theory.tex:28 writes s := (x_S, S), :33 the mask-aware Q(s, m, a) and :99 the
# aliasing Q(s, a), so the two implemented states are named after those.
METHOD_LABELS = {
    "aaco": "AACO",
    "aaco_doubly_robust": "AACO (doubly robust)",
    "dime": "DIME",
    "dime_feature_marginal_ipw": "DIME (feature-marginal IPW)",
    "gdfs": "GDFS",
    "jafa": "JAFA, $Q(s,a)$",
    "jafa_full_state": "JAFA, $Q(s,m,a)$",
    "ol_with_mask": "OL, $Q(s,a)$",
    "ol_full_state": "OL, $Q(s,m,a)$",
    "odin_model_free": "ODIN, $Q(s,a)$",
    "odin_model_free_full_state": "ODIN, $Q(s,m,a)$",
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

# Distinct per method, so two methods sharing a family hue still separate, and
# so identity survives greyscale printing.
METHOD_MARKERS = {
    "aaco": "o",
    "aaco_doubly_robust": "D",
    "dime": "^",
    "dime_feature_marginal_ipw": "v",
    "gdfs": "s",
    "jafa": "P",
    "jafa_full_state": "X",
    "ol_with_mask": "*",
    "ol_full_state": "h",
    "odin_model_free": "<",
    "odin_model_free_full_state": ">",
}

# Solid is a method, dashed is a reweighting control of a method.
METHOD_LINESTYLES = {
    "aaco": "solid",
    "aaco_doubly_robust": "dashed",
    "dime": "solid",
    "dime_feature_marginal_ipw": "dashed",
    "gdfs": "solid",
    "jafa": "solid",
    "jafa_full_state": "dashed",
    "ol_with_mask": "solid",
    "ol_full_state": "dashed",
    "odin_model_free": "solid",
    "odin_model_free_full_state": "dashed",
}

# prop:restriction gives Q_1^train = Q_1^eval and the inequality only at k >= 2,
# so the myopic/non-myopic split is the axis the theory is about. Greedy per the
# README taxonomy: GDFS and DIME estimate CMI one step at a time.
NON_MYOPIC_METHODS = frozenset(
    {
        "aaco",
        "aaco_doubly_robust",
        "aaco_nn",
        "jafa",
        "jafa_full_state",
        "ol_with_mask",
        "ol_without_mask",
        "ol_full_state",
        "odin_model_based",
        "odin_model_free",
        "odin_model_free_full_state",
    }
)

POLICY_TYPE_LINESTYLES = {"Myopic": "solid", "Non-myopic": "dotted"}


def policy_type(method: str) -> str:
    return "Non-myopic" if method in NON_MYOPIC_METHODS else "Myopic"


# Methods with both restricted-action and generative-restoration arms, in display
# order: the myopic pair first, then the non-myopic methods with each family's
# two Q states adjacent. The reweighting controls are trained on the restricted
# view only, so figures that compare training views omit them.
PRIMARY_METHODS = (
    "dime",
    "gdfs",
    "aaco",
    "jafa",
    "jafa_full_state",
    "ol_with_mask",
    "ol_full_state",
    "odin_model_free",
    "odin_model_free_full_state",
)

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
