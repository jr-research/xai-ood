"""The figures, from the emitted inputs and nothing else.

Four figures for the thesis and one appendix atlas. This script computes no statistic:
it reads counts, quantiles and AUROCs that `derive_distributions.py` and
`derive_spectral_panels.py` wrote after pinning themselves, and draws them. That
separation is deliberate and is kept, so that "the figure is wrong" and "the number is
wrong" stay different bugs in different files.

  fig6-score-overlap          ID against OOD, overlaid, two cells by six datasets. The
                              SVHN inversion in the naive cell is the panel that carries
                              it, and it replaces the mean, median and skew table that
                              stated the same thing in numbers.
  fig7-full-covariance-tail   The four factorial cells on clean ID, and where the excess
                              skew lives across the eight splits.
  fig8-rmd-cancellation       Two parents of order 768 and a difference of order 1, on
                              axes drawn to the same width.
  fig9-spectral-ablation      Cumulative share of the separation per eigendirection, and
                              AUROC against K.
  score-overlap-atlas.pdf     the per-method, per-dataset version of fig6 in full, all
                              thirteen scorers, one page each. An appendix object.

FORMAT. PDF for the thesis and PNG at 300 dpi for slides and notes, the same pair
`xai_ood.visualization.style.save_figure` writes and the same pair the rest of the
figure set ships in. The atlas is PDF only: it is thirteen pages and nothing pastes it into a
slide.

GREYSCALE. These print. No figure here uses hue as its only channel: the overlays
separate by fill against outline, the tail panel by position against a zero rule, the
cancellation by panel, and the ablation by line style.

Palette and sizes follow xai_ood.visualization.style. The module is not imported, because
this directory does not depend on the package; the constants are restated here and any
drift between the two is a bug in this file rather than in the package.

    python make_distribution_figures.py
"""

from __future__ import annotations

import csv
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "figures"

DPI = 300
SINGLE, DOUBLE, WIDE = (3.5, 2.6), (7.0, 3.0), (7.0, 4.5)

# Okabe-Ito, the canonical categorical palette, plus the two canonical neutrals.
ORANGE, SKY, GREEN = "#E69F00", "#56B4E9", "#009E73"
YELLOW, BLUE, VERMILLION = "#F0E442", "#0072B2", "#D55E00"
PURPLE, BLACK = "#CC79A7", "#000000"
GREY_DARK, GREY_MID = "#333333", "#666666"
GREY_LIGHT = "#7F7F7F"

SCORER_COLOR = {
    "marginal_diagonal": ORANGE,
    "marginal_full": SKY,
    "class_conditional_diagonal": GREEN,
    "class_conditional_full": BLUE,
    "rmd": VERMILLION,
    "rmd_pp": PURPLE,
    "knn_normalized": YELLOW,
    "knn_unnormalized": BLACK,
    "pca_residual_all_id": BLACK,
    "pca_residual_class_mean": GREEN,
    "pca_residual_all_id_l2": BLACK,
    "pca_residual_class_mean_l2": GREEN,
    "pca_residual_class_mean_whitened": GREEN,
}

DATASET_COLOR = {
    "cifar100": ORANGE, "tin": "#A6761D",
    "mnist": GREEN, "svhn": VERMILLION, "texture": BLUE, "places365": PURPLE,
}

LABEL = {
    "marginal_diagonal": "naive (no labels, diagonal)",
    "marginal_full": "no labels, full covariance",
    "class_conditional_diagonal": "class means, diagonal",
    "class_conditional_full": "MDS (class means, full)",
    "rmd": "RMD",
    "rmd_pp": "RMD, normalised",
    "knn_normalized": "kNN, normalised",
    "knn_unnormalized": "kNN, unnormalised",
    "pca_residual_all_id": "PCA residual, all-ID",
    "pca_residual_class_mean": "PCA residual, class-mean",
    "pca_residual_all_id_l2": "PCA residual, all-ID, L2",
    "pca_residual_class_mean_l2": "PCA residual, class-mean, L2",
    "pca_residual_class_mean_whitened": "PCA residual, whitened",
}
DATASET_LABEL = {
    "cifar100": "CIFAR-100", "tin": "Tiny ImageNet", "mnist": "MNIST",
    "svhn": "SVHN", "texture": "Texture", "places365": "Places365",
}
SPLIT_LABEL = dict(DATASET_LABEL, id_test="CIFAR-10 test (ID)", csid="CIFAR-10-C (cs-ID)")
GROUP_LABEL = {"near_ood": "near-OOD", "far_ood": "far-OOD"}

NEAR, FAR = ["cifar100", "tin"], ["mnist", "svhn", "texture", "places365"]
DATASETS = NEAR + FAR
SCORERS = list(LABEL)
FACTORIAL = ["marginal_diagonal", "marginal_full",
             "class_conditional_diagonal", "class_conditional_full"]

#: The row of fig6. The naive cell carries the inversion and the MDS cell is the one the
#: leaderboard is read off, so the two rows are the worst and the best of the 2x2.
OVERLAY_ROWS = ["marginal_diagonal", "class_conditional_full"]
PROTOCOL = "standard"


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,
    })


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        # CreationDate suppressed so a rebuild is byte-identical, the same reason
        # thesis-figures/make_figures.py gives.
        metadata = {"CreationDate": None} if suffix == "pdf" else {}
        fig.savefig(OUT / f"{name}.{suffix}", dpi=DPI, metadata=metadata)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


def load_histograms():
    return json.loads((HERE / "distribution-histograms.json").read_text())


def load_tail_shape():
    rows = []
    with (HERE / "tail-shape-by-cell-and-split.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    return rows


def centres(edges):
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


def density(counts, edges):
    """Counts to a density on the emitted grid. Not a statistic: a unit change."""
    counts = np.asarray(counts, dtype=float)
    width = float(edges[1] - edges[0])
    total = counts.sum()
    return counts / (total * width) if total else counts


# --------------------------------------------------------------------------- #
# Figure 6. ID against OOD, overlaid
#
# The claim: an AUROC is a summary of two overlapping distributions and the shape of that
# overlap is what the number cannot say. The naive cell's SVHN panel has the OOD mass
# entirely to the LEFT of the ID mass, which is an AUROC of 12.90 and is an inversion, not
# a failure to separate. Every other cell on that dataset has it on the right.
#
# Why overlaid and not side by side: separate panels make the reader superimpose them
# mentally, and superimposing is the whole content.
# --------------------------------------------------------------------------- #

def figure_overlap(hist):
    apply_style()
    fig, axes = plt.subplots(len(OVERLAY_ROWS), len(DATASETS),
                             figsize=(7.0, 3.4), sharex="row", sharey="row")
    for r, scorer in enumerate(OVERLAY_ROWS):
        entry = hist["scorers"][scorer]
        edges = np.asarray(entry["edges"], dtype=float)
        x = centres(edges)
        ident = density(entry["counts"]["id_pool_standard"], edges)
        for c, dataset in enumerate(DATASETS):
            ax = axes[r, c]
            ood = density(entry["counts"][dataset], edges)
            ax.fill_between(x, ident, color=GREY_LIGHT, alpha=0.75, lw=0, zorder=1)
            ax.plot(x, ood, color=DATASET_COLOR[dataset], lw=1.1, zorder=2)
            auroc = entry["auroc"][f"{PROTOCOL}/{dataset}"]
            inverted = auroc < 50.0
            ax.text(0.97, 0.94, f"{auroc:.2f}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=7,
                    color=VERMILLION if inverted else GREY_DARK,
                    fontweight="bold" if inverted else "normal")
            if inverted:
                ax.text(0.97, 0.76, "inverted", transform=ax.transAxes, ha="right",
                        va="top", fontsize=6.5, color=VERMILLION)
            if r == 0:
                ax.set_title(DATASET_LABEL[dataset], fontsize=8)
            if c == 0:
                ax.set_ylabel(LABEL[scorer], fontsize=7.5)
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=6.5)
            ax.spines["left"].set_visible(False)
    fig.supxlabel("score (squared Mahalanobis distance)", fontsize=8)
    fig.suptitle(
        "In-distribution (grey fill) against out-of-distribution (line), standard protocol, shrunk arm.\n"
        "Panel label is that cell's AUROC. Under the naive cell SVHN sits to the LEFT of ID: the ranking is inverted.",
        fontsize=8)
    save(fig, "fig6-score-overlap")


def atlas(hist):
    """All thirteen scorers, six datasets each. The appendix version of figure 6."""
    apply_style()
    OUT.mkdir(exist_ok=True)
    path = OUT / "score-overlap-atlas.pdf"
    with PdfPages(path, metadata={"CreationDate": None}) as pdf:
        for scorer in SCORERS:
            entry = hist["scorers"][scorer]
            edges = np.asarray(entry["edges"], dtype=float)
            x = centres(edges)
            ident = density(entry["counts"]["id_pool_standard"], edges)
            fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True)
            for ax, dataset in zip(axes.ravel(), DATASETS):
                ood = density(entry["counts"][dataset], edges)
                ax.fill_between(x, ident, color=GREY_LIGHT, alpha=0.75, lw=0)
                ax.plot(x, ood, color=DATASET_COLOR[dataset], lw=1.2)
                auroc = entry["auroc"][f"{PROTOCOL}/{dataset}"]
                ax.set_title(f"{DATASET_LABEL[dataset]}   AUROC {auroc:.2f}"
                             + ("   INVERTED" if auroc < 50 else ""), fontsize=8,
                             color=VERMILLION if auroc < 50 else GREY_DARK)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
            clipped = entry["clipped"]
            fig.suptitle(
                f"{LABEL[scorer]}  ({scorer})\n"
                f"grey fill: CIFAR-10 test, n = {entry['n']['id_pool_standard']}. "
                f"Line: the named OOD set. Standard protocol, shrunk arm.\n"
                f"Axis spans the 0.05th to 99.95th percentile of all rows for this scorer; "
                f"rows outside it are not drawn.", fontsize=8)
            fig.supxlabel("score", fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  wrote figures/{path.name} ({len(SCORERS)} pages)")


# --------------------------------------------------------------------------- #
# Figure 7. The full-covariance tail
#
# The claim, in two panels because it has two halves. Left: on clean in-distribution rows
# the two full-covariance cells carry a right tail and the two diagonal cells do not.
# Right: that excess is largest on the in-distribution rows, clean and corrupted alike,
# and is absent or reversed on far-OOD, so it is a property of the in-distribution
# population under whitening rather than of the score in general.
#
# Why excess and not raw skewness on the right: every split has its own shape, and MNIST's
# raw skew is -0.85 in all four cells. The quantity that isolates the design choice is the
# full cell minus its own diagonal counterpart on the same rows.
# --------------------------------------------------------------------------- #

def figure_tail(hist, tail_rows):
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.2))
    skew = {(r["arm"], r["scorer"], r["split"]): float(r["skewness"]) for r in tail_rows}

    style = {"marginal_diagonal": (":", 1.3), "class_conditional_diagonal": ("-.", 1.3),
             "marginal_full": ("--", 1.4), "class_conditional_full": ("-", 1.4)}
    for scorer in FACTORIAL:
        entry = hist["scorers"][scorer]
        edges = np.asarray(entry["edges"], dtype=float)
        dash, width = style[scorer]
        left.plot(centres(edges), density(entry["counts"]["id_test"], edges),
                  color=SCORER_COLOR[scorer], ls=dash, lw=width,
                  label=f"{LABEL[scorer]}, skew {skew[('shrunk', scorer, 'id_test')]:+.2f}")
    left.set_xlim(500, 2000)
    left.set_xlabel("score on the 9,000 clean ID images")
    left.set_ylabel("density")
    left.set_yticks([])
    left.spines["left"].set_visible(False)
    left.legend(frameon=False, fontsize=6.8, loc="upper right")
    left.set_title("Clean in-distribution scores", fontsize=8.5)

    order = ["id_test", "csid", "cifar100", "tin", "mnist", "svhn", "texture", "places365"]
    positions = np.arange(len(order))
    marginal = [skew[("shrunk", "marginal_full", s)] - skew[("shrunk", "marginal_diagonal", s)]
                for s in order]
    conditional = [skew[("shrunk", "class_conditional_full", s)]
                   - skew[("shrunk", "class_conditional_diagonal", s)] for s in order]
    right.axhline(0.0, color=GREY_MID, lw=0.8)
    right.bar(positions - 0.19, marginal, width=0.36, color=SKY, label="no labels")
    right.bar(positions + 0.19, conditional, width=0.36, color=BLUE, hatch="///",
              edgecolor="white", lw=0.0, label="class means")
    right.set_xticks(positions)
    right.set_xticklabels([SPLIT_LABEL[s] for s in order], rotation=40, ha="right", fontsize=6.5)
    right.set_ylabel("excess skewness")
    right.legend(frameon=False, fontsize=7, loc="upper right")
    right.set_title("Where the excess skew lives", fontsize=8.5)

    fig.suptitle(
        "The right tail tracks the covariance shape, not the class labels. Shrunk arm.\n"
        "Right: full cell minus its own diagonal counterpart on the same rows. The excess is\n"
        "largest on the in-distribution rows, clean and corrupted alike, and is at or below zero\n"
        "on far-OOD, with the one exception the panel shows.",
        fontsize=8)
    save(fig, "fig7-full-covariance-tail")


# --------------------------------------------------------------------------- #
# Figure 8. The RMD cancellation
#
# The claim: RMD is the difference of two quantities of order 768 and is itself of order 1,
# so almost all of each parent cancels. The picture is the two axes drawn at the same
# physical width with their own scales printed: the parents span roughly 1,500 score units
# and the difference spans roughly 100.
#
# Why it earns a figure: it is the mechanism behind a 0.86-point arm movement in a column
# whose parents move 0.003, and that is currently a paragraph of words on a slide.
# --------------------------------------------------------------------------- #

def figure_cancellation(hist):
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    target = "cifar100"

    for scorer, dash in (("marginal_full", "--"), ("class_conditional_full", "-")):
        entry = hist["scorers"][scorer]
        edges = np.asarray(entry["edges"], dtype=float)
        x = centres(edges)
        left.plot(x, density(entry["counts"]["id_test"], edges), color=SCORER_COLOR[scorer],
                  ls=dash, lw=1.3, label=f"{LABEL[scorer]}, ID")
        left.plot(x, density(entry["counts"][target], edges), color=SCORER_COLOR[scorer],
                  ls=dash, lw=1.3, alpha=0.45,
                  label=f"{LABEL[scorer]}, {DATASET_LABEL[target]}")
    parent_lo, parent_hi = 400.0, 2200.0
    left.set_xlim(parent_lo, parent_hi)
    left.set_xlabel("score")
    left.set_ylabel("density")
    left.set_yticks([])
    left.spines["left"].set_visible(False)
    left.legend(frameon=False, fontsize=6.2, loc="upper right")
    left.set_title(f"The two parents, over {parent_hi - parent_lo:,.0f} score units", fontsize=8.5)

    entry = hist["scorers"]["rmd"]
    edges = np.asarray(entry["edges"], dtype=float)
    x = centres(edges)
    right.fill_between(x, density(entry["counts"]["id_test"], edges), color=GREY_LIGHT,
                       alpha=0.75, lw=0, label="ID")
    right.plot(x, density(entry["counts"][target], edges), color=DATASET_COLOR[target],
               lw=1.3, label=DATASET_LABEL[target])
    right.set_xlabel("MDS minus no-labels full covariance, which is RMD")
    right.set_yticks([])
    right.spines["left"].set_visible(False)
    right.legend(frameon=False, fontsize=7, loc="upper right")
    right.set_title(f"Their difference, over {float(edges[-1]) - float(edges[0]):,.0f} units",
                    fontsize=8.5)
    right.text(0.03, 0.55, f"RMD, AUROC {entry['auroc'][f'{PROTOCOL}/{target}']:.2f}",
               transform=right.transAxes, fontsize=7, color=GREY_DARK)

    fig.suptitle(
        "RMD is a difference of near-equal terms. Both panels are the same width in inches;\n"
        "only the axis ranges differ. Clean ID and CIFAR-100, standard protocol, shrunk arm.",
        fontsize=8)
    save(fig, "fig8-rmd-cancellation")


# --------------------------------------------------------------------------- #
# Figure 9. Where the OOD signal sits along the spectrum
#
# The spectral decomposition's result, in the figure set's style.
# Left: cumulative share of the separation, in descending-eigenvalue order. Right: AUROC
# against K for "keep only the top K" and "discard the top K and keep the rest".
#
# ESTIMAND. Both group curves here are DATASET-AVERAGED, rebuilt from the
# decomposition's own per-dataset rows. Its group rows are pooled and differ by up to
# 0.48 AUROC points; `derive_spectral_panels.py` emits both and labels which is which.
# --------------------------------------------------------------------------- #

def figure_spectral():
    apply_style()
    payload = json.loads((HERE / "spectral-figure-input.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Colour is the second channel, never the only one: the group is carried by line
    # style on the left panel and by marker shape on the right, both of which survive
    # a greyscale print. Checked by rendering to luminance and re-reading.
    group_colour = {"near_ood": BLUE, "far_ood": VERMILLION}
    group_dash = {"near_ood": "-", "far_ood": "--"}
    group_marker = {"near_ood": "o", "far_ood": "s"}
    note = []
    for group, colour in group_colour.items():
        share = np.asarray(payload["groups"][group]["cumulative_share"], dtype=float)
        index = np.arange(1, share.size + 1)
        left.plot(index, 100.0 * share, lw=1.4, ls=group_dash[group], color=colour,
                  label=GROUP_LABEL[group])
        left.plot([256], [100.0 * share[255]], marker="o", ms=4, color=colour)
        note.append((colour, f"{GROUP_LABEL[group]}: the top 256 directions carry "
                             f"{100.0 * share[255]:+.1f}% of it"))
    left.axhline(100.0, color=GREY_MID, lw=0.8, ls=":")
    left.axhline(0.0, color=GREY_MID, lw=0.8)
    for row, (colour, text) in enumerate(note):
        left.text(0.03, 0.94 - 0.09 * row, text, transform=left.transAxes, fontsize=6.8,
                  color=colour, va="top")
    left.set_xscale("log")
    left.set_xlabel("eigendirections included, in descending eigenvalue order")
    left.set_ylabel("cumulative share of the separation (%)")
    left.legend(frameon=False, fontsize=7.5, loc="center left")
    left.set_title("Where the separation accumulates", fontsize=8.5)

    for group, colour in group_colour.items():
        entry = payload["groups"][group]
        forward = entry["forward_dataset_averaged"]
        backward = entry["backward_dataset_averaged"]
        ks = sorted(int(k) for k in forward)
        right.plot(ks, [forward[str(k)] for k in ks], ls="-", lw=1.3, color=colour,
                   marker=group_marker[group], ms=2.6, markevery=3,
                   label=f"{GROUP_LABEL[group]}, keep only the top K")
        right.plot(ks, [backward[str(k)] for k in ks], ls=":", lw=1.5, color=colour,
                   marker=group_marker[group], ms=2.6, markevery=3,
                   label=f"{GROUP_LABEL[group]}, discard the top K")
    right.axhline(50.0, color=GREY_MID, lw=0.8, ls=":")
    right.text(4, 52, "chance", fontsize=6.5, color=GREY_MID)
    right.set_xscale("log")
    right.set_xlabel("K, directions kept or discarded")
    right.set_ylabel("AUROC (%)")
    right.set_ylim(0, 102)
    right.legend(frameon=False, fontsize=6.2, loc="lower left")
    right.set_title("Keeping the top against discarding it", fontsize=8.5)

    fig.suptitle(
        "The OOD signal is in the low-variance directions. The no-labels full-covariance\n"
        "cell, shrunk arm, standard protocol. Group curves are dataset-averaged; the\n"
        "record carries the pooled companion and the gap between them.",
        fontsize=8)
    save(fig, "fig9-spectral-ablation")


def main() -> int:
    hist = load_histograms()
    tail_rows = load_tail_shape()
    print("DRAWING")
    figure_overlap(hist)
    figure_tail(hist, tail_rows)
    figure_cancellation(hist)
    figure_spectral()
    atlas(hist)
    print("\nWHAT THESE FIGURES DO NOT SHOW, printed every time:")
    for blind in (
        "any interval. No figure here draws uncertainty and none should be read as a test",
        "the unshrunk arm, except through the tail table that reports it",
        "the full-spectrum protocol, on which every drawn AUROC would move",
        "rows outside the 0.05th to 99.95th percentile of each scorer, which are clipped "
        "from the drawn axis and counted in distribution-histograms.json",
    ):
        print(f"  - {blind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
