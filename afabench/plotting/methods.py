"""Shared colors, labels, and plotting style."""

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
    "dime": "dime",
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
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": INK,
            "ytick.color": INK,
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
    "dime": "DIME",
    "gdfs": "GDFS",
    "jafa": "JAFA, $Q(s,a)$",
    "jafa_full_state": "JAFA, $Q(s,m,a)$",
    "ol_with_mask": "OL, $Q(s,a)$",
    "ol_full_state": "OL, $Q(s,m,a)$",
    "odin_model_free": "ODIN, $Q(s,a)$",
    "odin_model_free_full_state": "ODIN, $Q(s,m,a)$",
}

METHOD_LABELS_SHORT = {
    "aaco": "AACO",
    "dime": "DIME",
    "gdfs": "GDFS",
    "jafa": r"JAFA $s$",
    "jafa_full_state": r"JAFA $s{,}m$",
    "ol_with_mask": r"OL $s$",
    "ol_full_state": r"OL $s{,}m$",
    "odin_model_free": r"ODIN $s$",
    "odin_model_free_full_state": r"ODIN $s{,}m$",
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
    "dime": "^",
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
    "dime": "solid",
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


# Methods in display order, with state variants adjacent.
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
