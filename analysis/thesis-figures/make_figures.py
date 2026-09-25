"""The five thesis figures, from figure-inputs.json and nothing else.

Every number drawn here comes from that file, which refuses to exist unless its derivation
first reproduced a gated table. This script computes no statistic: it reads points and
intervals and draws them. That separation is deliberate, so that "the figure is wrong"
and "the number is wrong" are different bugs in different files.

Palette and sizes follow xai_ood.visualization.style: Okabe-Ito, DejaVu Sans at 9 pt,
3.5 in single column and 7.0 in double, PDF plus PNG at 300 dpi. The module is not
imported, because this directory does not depend on the package; the constants are
restated and any drift is a bug in this file, which is what
`../figure-style-check/check_style.py` exists to catch.

GREYSCALE. These print. Every figure must survive losing its colour, so no figure uses
hue as its only channel: the slope chart separates by weight and direct labelling, the
interaction plot by marker and line style, the dot plot by position alone, the interval
plot by position against a zero rule, and the dendrogram by geometry. Colour is a second
channel on all five and the only channel on none.

    ~/.venvs/xai-ood-parquet/bin/python make_figures.py
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

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

# Prose names. A thesis figure does not print a Python identifier at a reader.
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
GROUP_LABEL = {"near_ood": "near-OOD", "far_ood": "far-OOD"}

NEAR, FAR = ["cifar100", "tin"], ["mnist", "svhn", "texture", "places365"]


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.constrained_layout.use": True,
        "pdf.fonttype": 42,  # embed as TrueType, so the PDF's text stays selectable
    })


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        # CreationDate suppressed so a rebuild is byte-identical. matplotlib stamps the
        # wall clock into PDF metadata by default, which made every regeneration dirty
        # five tracked files and would have turned "re-run the figures before submitting"
        # into a diff nobody reads.
        metadata = {"CreationDate": None} if suffix == "pdf" else {}
        fig.savefig(OUT / f"{name}.{suffix}", dpi=DPI, metadata=metadata)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")


# --------------------------------------------------------------------------- #
# Figure 1. Degradation as a slope chart
#
# The claim: detecting new categories and staying reliable when input quality drops are
# different properties. RMD is near the top on clean data and near the bottom under
# covariate shift, and three detectors improve.
#
# Why a slope and not a bar of differences: the crossing is the finding. A bar chart of
# "AUROC lost" sorts RMD to one end and says nothing about where it started, so the reader
# cannot see that the worst degrader was among the best clean detectors.
# --------------------------------------------------------------------------- #

HIGHLIGHT = ["rmd", "rmd_pp"]
IMPROVERS = ["marginal_diagonal", "class_conditional_diagonal", "pca_residual_class_mean"]


def _spread(anchors, minimum_gap):
    """Nudge label anchors apart so end-of-line labels do not overprint each other.

    Direct labelling is what makes this figure survive greyscale, so a collision is not
    cosmetic: two labels on top of each other put the reader back on a legend. The nudge
    moves only the TEXT; every line still ends at its true value, and the marker is drawn
    at the value rather than at the label.
    """
    order = sorted(range(len(anchors)), key=lambda i: anchors[i])
    placed = list(anchors)
    for position, index in enumerate(order):
        if position == 0:
            continue
        previous = placed[order[position - 1]]
        if placed[index] - previous < minimum_gap:
            placed[index] = previous + minimum_gap
    return placed


def figure_degradation(data):
    fig, axes = plt.subplots(1, 2, figsize=WIDE, sharey=True)
    for ax, group in zip(axes, ("near_ood", "far_ood")):
        labelled = []
        for scorer, per_group in data["degradation"].items():
            values = per_group[group]
            y = [values["standard"], values["full_spectrum"]]
            if scorer in HIGHLIGHT:
                kw = dict(color=SCORER_COLOR[scorer], lw=2.0, zorder=3, marker="o", ms=4)
            elif scorer in IMPROVERS:
                kw = dict(color=SCORER_COLOR[scorer], lw=1.3, ls="--", zorder=2,
                          marker="s", ms=3)
            else:
                kw = dict(color=GREY_LIGHT, lw=0.9, zorder=1, marker="o", ms=2.5)
            ax.plot([0, 1], y, **kw)
            if scorer in HIGHLIGHT + IMPROVERS:
                labelled.append((scorer, y[1]))
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        for (scorer, end), text_y in zip(labelled, _spread([e for _, e in labelled],
                                                           0.042 * span)):
            ax.annotate(LABEL[scorer], xy=(1.04, text_y), va="center", fontsize=7,
                        color=SCORER_COLOR[scorer])
            if abs(text_y - end) > 1e-9:
                ax.plot([1.0, 1.035], [end, text_y], color=SCORER_COLOR[scorer],
                        lw=0.6, ls="-", zorder=1)
        ax.set_xlim(-0.12, 1.85)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["clean", "covariate shift"])
        ax.set_title(GROUP_LABEL[group])
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("AUROC (%)")
    fig.suptitle("Every detector, clean protocol to covariate shift", fontsize=9.5)
    save(fig, "fig1-degradation-slope")


# --------------------------------------------------------------------------- #
# Figure 2. The 2x2 as an interaction plot
#
# The claim: the two Gaussian mechanisms are substitutes, not complements. The second one
# to arrive finds almost nothing left to do.
#
# Why a figure and not the table: the finding IS the difference between two gaps, so a
# table asks the reader to subtract twice and then compare. Non-parallel lines say it at a
# glance, and the near-parallel-to-flat upper line under covariate shift is the whole point.
# --------------------------------------------------------------------------- #

def figure_interaction(data):
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE, sharey=True)
    cells = data["factorial"]["cells"]
    gaps = data["factorial"]["full_minus_diagonal"]
    # Two levels of one factor, so they need to read apart in hue AND in line style: the
    # canonical per-scorer colours for these two rows are both blues, which is fine when a
    # figure plots one family and not fine when it plots both on one axis. The style
    # module's own note prescribes line style as the tiebreak; this takes the marginal
    # row's diagonal-cell colour as well, so the naive cell keeps the colour it has in
    # every other figure here.
    series = {
        "marginal": dict(color=ORANGE, ls="-", marker="o", label="no class labels"),
        "class_conditional": dict(color=BLUE, ls="--", marker="s",
                                  label="class means used"),
    }
    for ax, protocol in zip(axes, ("standard", "full_spectrum")):
        for labels, style in series.items():
            points = [cells[protocol]["near_ood"][f"{labels}|{c}"]
                      for c in ("diagonal", "full")]
            ax.errorbar([0, 1], [c["point"] for c in points],
                        yerr=[[c["point"] - c["lo"] for c in points],
                              [c["hi"] - c["point"] for c in points]],
                        capsize=2, lw=1.7, ms=4.5, color=style["color"],
                        ls=style["ls"], marker=style["marker"])
            # The gap annotation sits at the segment's midpoint, which is empty space for
            # the steep line and just above the flat one, rather than at a hand-picked
            # coordinate that stops being right the moment a number moves.
            gap = gaps[protocol]["near_ood"][labels]
            mid_y = (points[0]["point"] + points[1]["point"]) / 2
            below = labels == "marginal"
            text = f"{gap['point']:+.2f}\n[{gap['lo']:+.2f}, {gap['hi']:+.2f}]"
            # An average whose two datasets fall on opposite sides of zero says something
            # different from an average near zero, and the figure has to say which.
            # Without this line the covariate-shift cell reads "+0.04, indistinguishable
            # from zero" when it is CIFAR-100 at -0.44 and Tiny ImageNet at +0.53, both
            # excluding zero. check_figure_claims.py is what found it.
            if not gap["datasets_agree_in_sign"]:
                parts = ", ".join(f"{DATASET_LABEL[d]} {c['point']:+.2f}"
                                  for d, c in gap["per_dataset"].items())
                text += f"\nbut {parts}"
            ax.annotate(text,
                        xy=(0.5, mid_y), xytext=(0, -14 if below else 12),
                        textcoords="offset points", ha="center",
                        va="top" if below else "bottom",
                        fontsize=7, color=style["color"])
            ax.annotate(style["label"], xy=(0, points[0]["point"]), xytext=(-3, -10),
                        textcoords="offset points", ha="left", va="top",
                        fontsize=7.5, color=style["color"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["diagonal", "full covariance"])
        ax.set_title({"standard": "clean protocol",
                      "full_spectrum": "covariate shift"}[protocol])
    axes[0].set_ylabel("AUROC (%), near-OOD")
    axes[0].set_ylim(68, 107)  # headroom for the flat line's gap annotation
    fig.suptitle("Adding full covariance is worth far less once class means are used",
                 fontsize=9.5)
    save(fig, "fig2-factorial-interaction")


# --------------------------------------------------------------------------- #
# Figure 3. The per-dataset breakdown, and the inversion the average hides
#
# The claim: the naive cell does not fail to separate on SVHN, it separates confidently in
# the wrong direction, and the far-OOD average of 60.89 is the mean of two opposite
# behaviours.
#
# Why a figure and not the table: the chance line is the argument, and a table cannot draw
# one. Position against 50 does the work, so this figure is readable with no colour at all.
# --------------------------------------------------------------------------- #

def figure_per_dataset(data):
    fig, axes = plt.subplots(1, 2, figsize=DOUBLE, sharey=True)
    order = NEAR + FAR
    # Only the left panel is annotated. The right panel exists to show that the same six
    # datasets land inside two AUROC points once the model has labels and a covariance,
    # and its individual numbers are unreadable at that spacing and are not the claim.
    # Labelling both panels was tried and produced a figure with two arguments and one
    # legible one; the per-dataset numbers belong in an appendix table.
    for ax, scorer, annotate in ((axes[0], "marginal_diagonal", True),
                                 (axes[1], "class_conditional_full", False)):
        block = data["per_dataset_standard"][scorer]
        for i, dataset in enumerate(order):
            c = block["datasets"][dataset]
            y = len(order) - 1 - i
            ax.errorbar(c["point"], y,
                        xerr=[[c["point"] - c["lo"]], [c["hi"] - c["point"]]],
                        fmt="o", ms=4.5, capsize=2, lw=1.2,
                        color=DATASET_COLOR[dataset])
            if annotate:
                ax.annotate(f"{c['point']:.2f}", xy=(c["point"], y), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=7,
                            color=GREY_DARK)
        ax.axvline(50, color=GREY_DARK, lw=1.0, ls=":")
        for group, members in (("near_ood", NEAR), ("far_ood", FAR)):
            mean = block["group_mean"][group]["point"]
            rows = [len(order) - 1 - order.index(d) for d in members]
            ax.plot([mean, mean], [min(rows) - 0.35, max(rows) + 0.35],
                    color=GREY_MID, lw=1.4)
            if annotate:
                # Rotated along its own rule. A horizontal label here collides with
                # whichever dot sits near the group mean, and the collision moves as the
                # numbers move.
                # Short. Rotated text as long as "far-OOD mean 60.89" runs past the
                # four-row far group and off the bottom of the axes.
                ax.annotate(f"{GROUP_LABEL[group]} {mean:.2f}",
                            xy=(mean, (min(rows) + max(rows)) / 2), xytext=(-4, 0),
                            textcoords="offset points", fontsize=6.8, rotation=90,
                            ha="right", va="center", rotation_mode="anchor",
                            color=GREY_MID)
        if not annotate:
            values = [block["datasets"][d]["point"] for d in order]
            ax.annotate(f"all six datasets between {min(values):.2f} and {max(values):.2f};\n"
                        f"the two group means differ by "
                        f"{abs(block['group_mean']['near_ood']['point'] - block['group_mean']['far_ood']['point']):.2f}",
                        xy=(50, 1.2), ha="center", va="center", fontsize=7,
                        color=GREY_DARK)
        ax.set_xlim(0, 108)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([DATASET_LABEL[d] for d in reversed(order)])
        ax.set_ylim(-0.9, len(order) - 0.3)
        ax.set_title(LABEL[scorer])
        ax.set_xlabel("AUROC (%)")
        ax.annotate("chance", xy=(50, -0.8), xytext=(-3, 0), textcoords="offset points",
                    fontsize=6.8, ha="right", va="bottom", color=GREY_DARK)
    fig.suptitle("Two datasets sit below chance, and the group average hides it",
                 fontsize=9.5)
    save(fig, "fig3-per-dataset-inversion")


# --------------------------------------------------------------------------- #
# Figure 4. The arm comparison, per dataset
#
# The claim: the near-OOD sign change between covariance arms lives in the average, not in
# either dataset. CIFAR-100 favours RMD on both arms and Tiny ImageNet favours MDS on both
# arms; the two disagree by 2.71 and 2.19 points, and the arm moves the average by 0.86.
#
# WHY THIS REPLACES AN EARLIER DRAFT, and the earlier one should not be revived. A figure
# of the two dataset-averaged contrasts against a zero rule was built first, and it is a
# picture of an averaging artefact presented as a finding about the covariance estimator.
# Two non-overlapping intervals on opposite sides of zero is a compelling image, and it
# was compelling about the wrong thing: the mean of two numbers that straddle zero crosses
# zero under any shift smaller than the gap between them, which is what happened.
#
# The honest version has to show what the average is an average OF, so this one draws the
# four dataset cells and puts the two averages below a rule, as the artefact rather than
# as the result.
# --------------------------------------------------------------------------- #

def figure_arm_comparison(data):
    block = data["arm_comparison_auroc"]
    rows = [
        ("tin", "shrunk", "Tiny ImageNet, shrunk"),
        ("tin", "unshrunk", "Tiny ImageNet, unshrunk"),
        ("cifar100", "shrunk", "CIFAR-100, shrunk"),
        ("cifar100", "unshrunk", "CIFAR-100, unshrunk"),
        (None, "shrunk", "near-OOD average, shrunk"),
        (None, "unshrunk", "near-OOD average, unshrunk"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for i, (dataset, arm, label) in enumerate(rows):
        y = len(rows) - 1 - i
        if dataset is None:
            c = block["averages"][arm]
            color, marker, fill = GREY_MID, "s", "none"
        else:
            c = block["datasets"][dataset][arm]
            color, marker, fill = DATASET_COLOR[dataset], "o", None
        ax.errorbar(c["point"], y,
                    xerr=[[c["point"] - c["lo"]], [c["hi"] - c["point"]]],
                    fmt=marker, ms=5, capsize=3, lw=1.6, color=color,
                    markerfacecolor=fill if fill else color)
        ax.annotate(f"{c['point']:+.4f}  [{c['lo']:+.4f}, {c['hi']:+.4f}]",
                    xy=(c["point"], y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7, color=GREY_DARK)
    ax.axvline(0, color=BLACK, lw=1.1)
    # The rule sits above the averages' own value labels, not at the midpoint between
    # rows, which is where a label lands.
    ax.axhline(1.72, color=GREY_LIGHT, lw=0.6, ls=":")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for _, _, label in reversed(rows)])
    ax.set_ylim(-2.35, len(rows) - 0.25)
    ax.set_xlim(-2.1, 3.1)
    ax.set_xlabel("MDS minus RMD, AUROC (%), near-OOD, clean protocol")
    ax.set_title("The sign change is in the average, not in either dataset", fontsize=9.5)
    gaps = block["between_dataset_gap"]
    ax.annotate(
        f"The two datasets disagree by {gaps['shrunk']:.2f} points on the shrunk fit and "
        f"{gaps['unshrunk']:.2f} on the unshrunk one, and neither\n"
        f"changes sign. The fit moves the average by "
        f"{block['average_moves_between_arms']:.2f}, across a zero it already sat on.\n"
        "Dataset-averaged estimand, B = 1,000. Intervals are descriptive.",
        xy=(-2.05, -1.45), fontsize=6.8, color=GREY_MID, va="center", ha="left")
    save(fig, "fig4-arm-comparison-per-dataset")


# --------------------------------------------------------------------------- #
# Figure 5. How many distinct detectors there really are
#
# The claim: thirteen configurations are not thirteen independent pieces of evidence.
#
# Why a dendrogram and not a heatmap: a 13x13 heatmap is 169 cells a reader scans for
# structure, it needs colour to carry a magnitude, and it is unreadable in greyscale and at
# seminar distance. A dendrogram answers the actual question, "which of these are the same
# detector", with geometry, and its merge heights are the correlations.
#
# scipy earns its place here: average linkage over a condensed distance matrix and the
# drawn tree are ten correct lines against roughly forty hand-rolled ones whose merge
# heights nothing would check. It is already in the analysis environment, so this is not a
# new dependency, and the linkage API has been stable for over a decade.
# --------------------------------------------------------------------------- #

def figure_redundancy(data):
    block = data["rank_correlations"]
    names = block["order"]
    rho = np.asarray(block["matrix"], dtype=float)
    # Distance is 1 - |rho|: two scorers that rank OOD images in opposite order are as
    # redundant as two that agree, because one is the other negated. RMD against the naive
    # cell is the case that makes this the right choice rather than a convention.
    distance = 1.0 - np.abs(rho)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    tree = linkage(squareform(distance, checks=False), method="average")

    fig, ax = plt.subplots(figsize=WIDE)
    dendrogram(tree, labels=[LABEL[n] for n in names], orientation="right",
               color_threshold=0.0, above_threshold_color=GREY_DARK, ax=ax)
    ax.set_xlabel("1 − |Spearman ρ| between per-image scores, OOD rows")
    ax.set_title("Thirteen configurations, clustered by how they rank OOD images",
                 fontsize=9.5)
    # Deliberately NO cut line at eleven clusters. The count of eleven distinct detectors
    # came from a six-stage audit whose own README says a correlation matrix is a screen
    # and not a verdict; a threshold drawn here would claim the number fell out of this
    # clustering, which it did not, and a reader would then be able to move it by moving
    # the threshold. The figure is the screen. The caption carries the count.
    ax.annotate("Rank agreement only. The count of eleven distinct detectors comes from\n"
                "the six-stage redundancy audit, not from a cut through this tree.",
                xy=(0.30, 0.45), xycoords="axes fraction", fontsize=6.8, color=GREY_MID)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, "fig5-scorer-redundancy")


def main():
    with open(HERE / "figure-inputs.json") as handle:
        data = json.load(handle)
    print("provenance:", json.dumps(data["_provenance"], indent=1))
    apply_style()
    figure_degradation(data)
    figure_interaction(data)
    figure_per_dataset(data)
    figure_arm_comparison(data)
    figure_redundancy(data)


if __name__ == "__main__":
    main()
