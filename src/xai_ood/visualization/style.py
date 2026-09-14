"""The one figure style for the whole thesis.

Decide font, size, colormap and the AUROC scale once, put them in a single
helper, and never re-export twenty figures for consistency late on. Every
plotting function imports from here.

**The decisions, so they are findable in one place:**

- **AUROC and FPR@95 are on the 0-100 scale.** ``98.46``, not ``0.9846``. This
  matches OpenOOD's published tables, which is what the numbers are compared
  against.
  ``xai_ood.schema.validate_results_frame`` enforces the same scale on the table
  side, so a 0-1 value cannot reach a figure axis unnoticed.
- **Categorical palette: Okabe-Ito**, the eight-colour colourblind-safe set from
  Okabe & Ito (2008). Distinguishable under deuteranopia, protanopia and
  tritanopia, which the matplotlib default cycle is not.
- **Sequential colormap: viridis**, for anything ordered -- severity curves above
  all, plus covariance heatmaps and eigenspectra.
- **Font: DejaVu Sans at 9 pt.** Deliberately matplotlib's bundled default rather
  than Helvetica or Arial, so a figure renders identically on any machine with
  matplotlib installed, without a font-availability gamble.
- **Sizes: 3.5 in single column, 7.0 in double.** Figures are placed at their
  natural size, never rescaled in LaTeX, so 9 pt in the figure is 9 pt on the page.
- **Output: PDF (vector) for the thesis, PNG at 300 dpi for slides and notes.**
  ``save_figure`` writes both from one call.

Nothing here is expensive to change now, and all of it is expensive to change
once figures have been exported. That asymmetry is the entire reason this file
exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "AUROC_SCALE",
    "AUROC_LABEL",
    "FPR95_LABEL",
    "OKABE_ITO",
    "DATASET_COLORS",
    "PALETTE_EXTENSIONS",
    "NEUTRALS",
    "SEQUENTIAL_CMAP",
    "DIVERGING_CMAP",
    "BASE_FONT_SIZE",
    "SINGLE_COLUMN",
    "DOUBLE_COLUMN",
    "WIDE",
    "DPI",
    "format_auroc",
    "to_display_scale",
    "series_color",
    "dataset_color",
    "apply_style",
    "new_figure",
    "save_figure",
    "severity_colors",
]


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #

#: AUROC and FPR@95 are reported on 0-100 everywhere: tables, figures, prose.
AUROC_SCALE: float = 100.0

AUROC_LABEL: str = "AUROC (%)"
FPR95_LABEL: str = "FPR@95 (%)"


def to_display_scale(value: Any) -> Any:
    """Convert a raw 0-1 metric to the 0-100 display scale.

    Use this at exactly one point: where a metric from ``xai_ood.metrics``,
    which returns 0-1, enters the results table. Everything
    downstream of the table is already on 0-100 and must not be scaled again.
    Works on scalars, numpy arrays and pandas Series alike.
    """
    return value * AUROC_SCALE


def format_auroc(value: Any, decimals: int = 2) -> str:
    """Format a 0-100-scale metric for a table cell or an annotation."""
    if value is None:
        return "--"
    try:
        if value != value:  # NaN, without importing numpy
            return "--"
    except TypeError:
        return "--"
    return f"{float(value):.{decimals}f}"


# --------------------------------------------------------------------------- #
# Colour
# --------------------------------------------------------------------------- #

#: Okabe & Ito (2008) colourblind-safe qualitative palette. Order is the
#: published one; black is deliberately last so it is not the default first draw.
OKABE_ITO: tuple[str, ...] = (
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#000000",  # black
)

SEQUENTIAL_CMAP: str = "viridis"
DIVERGING_CMAP: str = "coolwarm"

#: Stable colour per named series, so a scorer keeps the same colour in every
#: figure in the thesis. Names not listed here fall back to a hash-stable slot,
#: which is still deterministic across runs and machines.
_SERIES_COLOR: dict[str, str] = {
    # 2x2 Gaussian factorial
    "marginal_diagonal": OKABE_ITO[0],  # the naive baseline
    "marginal_full": OKABE_ITO[1],
    "class_conditional_diagonal": OKABE_ITO[2],
    "class_conditional_full": OKABE_ITO[4],  # MDS
    # relative variants
    "rmd": OKABE_ITO[5],
    "rmd_pp": OKABE_ITO[6],
    # The two normalised Full cells, added 2026-09-13 and post-hoc. Each takes
    # its unnormalised parent's colour rather than a fresh slot, on the same
    # rule the PCA-residual variants follow below: a variant stays visually
    # attached to the cell it varies, so a reader sees a normalisation pair
    # rather than two unrelated series.
    "marginal_full_pp": OKABE_ITO[1],
    "class_conditional_full_pp": OKABE_ITO[4],
    # other families
    "knn_normalized": OKABE_ITO[3],  # Sun et al.'s canonical form, so it keeps
    "knn_unnormalized": OKABE_ITO[7],  # the family colour; the ablation differs
    "pca_residual_all_id": OKABE_ITO[7],
    "pca_residual_class_mean": OKABE_ITO[2],
    # The PCA-residual variants. Each takes its *subspace's* colour rather than
    # a new slot, so the two PCA-residual components stay visually distinct from
    # each other while a variant stays visually attached to the component it
    # varies. See the note below for why they are not given fresh colours.
    "pca_residual_all_id_l2": OKABE_ITO[7],
    "pca_residual_class_mean_l2": OKABE_ITO[2],
    "pca_residual_class_mean_whitened": OKABE_ITO[2],
    # There is deliberately no `pca_residual_all_id_whitened` entry:
    # `PcaResidualConfig` refuses that combination outright, because whitening
    # makes the all-ID cloud's covariance the identity and leaves a
    # round-off-determined subspace. This dict is a registry of series that
    # exist; an entry for something that cannot be built is a false affordance.
}

# NOTE: there are more named series than the eight Okabe-Ito slots, so some
# collide -- knn_unnormalized with pca_residual_all_id, and
# class_conditional_diagonal with pca_residual_class_mean. Harmless while each
# figure plots one family, which is how the severity curve and the 2x2 panel are
# planned. If a figure ever puts
# colliding series on the same axes, distinguish them by line style rather than
# by re-assigning colours here: re-assigning would silently recolour every
# figure already exported, which is the exact cost this module exists to avoid.
#
# The PCA-residual family's four variants deliberately share their subspace's
# colour, which makes the line-style rule above **load-bearing rather than
# contingent** for one planned figure. The normalization ablation
# puts pca_residual_all_id against pca_residual_all_id_l2, and the degeneracy
# remedy puts pca_residual_class_mean against pca_residual_class_mean_whitened --
# same colour, same axes, by design. Those figures must distinguish the primary
# from its variant by line style (solid for the primary, dashed for the
# ablation, dotted for the whitened remedy). The alternative was four more
# colours from an eight-colour palette that is already over-subscribed, which
# would have meant either re-assigning existing series or leaving the
# colourblind-safe set, and the whole point of the primary/variant pairing is
# that the two *should* read as the same series under two treatments.


#: Stable colour per dataset split. A separate registry from ``_SERIES_COLOR``
#: because the two are keyed on different things and a figure never mixes them:
#: one plots scorers, the other plots the data those scorers were run on.
#:
#: **Two of these are deliberately outside Okabe-Ito, and that is a considered
#: deviation rather than drift.** There are ten splits and eight slots, and the
#: two that leave the set are the two that should not compete for attention: the
#: held-out in-distribution split, drawn neutral grey, and Tiny ImageNet, whose
#: brown is taken from Colorbrewer Dark2 and stays distinguishable under all
#: three common forms of colour blindness. Registering them here rather than
#: leaving them in a plotting script is the point of this block: the deviation is
#: now visible in the place someone would look for it.
#:
#: ``cifar10_train`` and ``cifar10_test`` share black on purpose. They are the
#: same distribution under two roles, and no planned figure draws both.
DATASET_COLORS: dict[str, str] = {
    "cifar10_train": OKABE_ITO[7],   # black
    "cifar10_test": OKABE_ITO[7],    # black, same distribution as train
    "cifar10_val": "#7F7F7F",        # neutral grey, outside the palette
    "csid": OKABE_ITO[1],            # sky blue, the covariate-shifted grid
    "cifar100": OKABE_ITO[0],        # orange, near-OOD
    "tin": "#A6761D",                # brown, outside the palette, near-OOD
    "mnist": OKABE_ITO[2],           # bluish green, far-OOD
    "svhn": OKABE_ITO[5],            # vermillion, far-OOD
    "texture": OKABE_ITO[4],         # blue, far-OOD
    "places365": OKABE_ITO[6],       # reddish purple, far-OOD
}

#: Colours in use that are not from Okabe-Ito, named so a checker can tell a
#: registered deviation apart from an unregistered one.
PALETTE_EXTENSIONS: tuple[str, ...] = ("#7F7F7F", "#A6761D")

#: Neutral greys for annotation, captions and secondary axis text. These are not
#: data colours and must never encode a variable. They are declared here because
#: three separate figure scripts had each invented their own grey, one of them
#: four different ones in a single file, which is the same drift the palette
#: block exists to prevent and was invisible while no registry named them.
NEUTRALS: tuple[str, ...] = (
    "#333333",  # body and axis text
    "#666666",  # captions, secondary annotation, de-emphasised marks
)


def dataset_color(name: str) -> str:
    """Colour for a named dataset split, stable across figures and runs.

    Same deterministic fallback as :func:`series_color`, so an unregistered
    split still gets the same colour on every machine and every run.
    """
    if name in DATASET_COLORS:
        return DATASET_COLORS[name]
    slot = sum(ord(c) for c in name) % len(OKABE_ITO)
    return OKABE_ITO[slot]


def series_color(name: str) -> str:
    """Colour for a named scorer or series, stable across figures and runs."""
    if name in _SERIES_COLOR:
        return _SERIES_COLOR[name]
    # Deterministic fallback: sum of code points, not hash(), because Python's
    # str hash is salted per-process and would give a different colour each run.
    slot = sum(ord(c) for c in name) % len(OKABE_ITO)
    return OKABE_ITO[slot]


# --------------------------------------------------------------------------- #
# Geometry and type
# --------------------------------------------------------------------------- #

BASE_FONT_SIZE: float = 9.0

SINGLE_COLUMN: tuple[float, float] = (3.5, 2.6)
DOUBLE_COLUMN: tuple[float, float] = (7.0, 3.0)
WIDE: tuple[float, float] = (7.0, 4.5)  # severity grids, heatmaps

DPI: int = 300


def apply_style() -> None:
    """Install the thesis style into matplotlib's global rcParams.

    Call once at the top of any plotting script. matplotlib is imported lazily so
    this module stays importable (and testable) in an environment without it.
    """
    import matplotlib as mpl
    from cycler import cycler

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": BASE_FONT_SIZE,
            "axes.titlesize": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "xtick.labelsize": BASE_FONT_SIZE - 1,
            "ytick.labelsize": BASE_FONT_SIZE - 1,
            "legend.fontsize": BASE_FONT_SIZE - 1,
            "figure.titlesize": BASE_FONT_SIZE + 1,
            "figure.figsize": SINGLE_COLUMN,
            "figure.dpi": 120,          # on-screen
            "savefig.dpi": DPI,         # on-disk raster
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": False,
            "axes.prop_cycle": cycler(color=list(OKABE_ITO)),
            "image.cmap": SEQUENTIAL_CMAP,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "legend.frameon": False,
            "pdf.fonttype": 42,  # embed TrueType, keeps text selectable in the PDF
            "ps.fonttype": 42,
        }
    )


def new_figure(size: tuple[float, float] = SINGLE_COLUMN, **kwargs: Any):
    """``plt.subplots`` with the thesis style applied and a standard size."""
    apply_style()
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=size, **kwargs)


def save_figure(fig: Any, path: str | Path, *, formats: Sequence[str] = ("pdf", "png")) -> list[Path]:
    """Save ``fig`` to ``path`` in every requested format; return the paths written.

    ``path`` may carry an extension or not; it is stripped either way and one file
    per entry in ``formats`` is written. Parent directories are created.
    """
    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = base.with_suffix(f".{fmt}")
        fig.savefig(out, format=fmt)
        written.append(out)
    return written


def severity_colors(n: int = 5) -> list[str]:
    """``n`` colours sampled from viridis, for severity-ordered series.

    Severity is ordered, so it gets the sequential map rather than the
    categorical palette. Scorers get ``series_color``; severities get this.
    """
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(SEQUENTIAL_CMAP)
    import matplotlib.colors as mcolors

    return [mcolors.to_hex(cmap(i / max(n - 1, 1))) for i in range(n)]
