"""First look at the score distributions, which every table so far reduced away.

Produced 2026-09-12 after a mean and a median disagreed by 27.7 points on the
full-covariance cell while agreeing on the diagonal one. The figure is the point: a
right tail that is invisible in any summary statistic is obvious in a histogram.

Read-only against the run directories. Writes two PNGs beside this script.
"""
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

# The repository root, the directory above this one. Overridable with
# XAI_OOD_ROOT so a run tree held outside the checkout can be read in place.
ROOT = pathlib.Path(os.environ.get("XAI_OOD_ROOT",
                                   pathlib.Path(__file__).resolve().parents[1]))
RUN = ROOT / "results" / "runs" / "2026-09-10-full-spectrum-shrunk" / "scores"
OUT = pathlib.Path(__file__).resolve().parent
CELLS = ["marginal_diagonal", "marginal_full",
         "class_conditional_diagonal", "class_conditional_full"]
DIM = 768  # a squared Mahalanobis distance of a sample from its own fit has expectation = dim


def load(columns):
    return pq.read_table(RUN / "per_sample_scores.parquet", columns=columns).to_pandas()


def figure_one(frame):
    """The four factorial cells on clean ID rows. Diagonal cells symmetric, full cells skewed."""
    ident = frame[frame.split == "id_test"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, cell in zip(axes.ravel(), CELLS):
        values = ident[cell].to_numpy(float)
        mean, median = values.mean(), np.median(values)
        skew = ((values - mean) ** 3).mean() / values.std() ** 3
        ax.hist(values, bins=120, color="#4C72B0", alpha=0.85)
        ax.axvline(median, color="#C44E52", lw=1.4, label=f"median {median:.0f}")
        ax.axvline(mean, color="#DD8452", lw=1.4, ls="--", label=f"mean {mean:.0f}")
        ax.axvline(DIM, color="0.35", lw=1.0, ls=":", label=f"dimension {DIM}")
        ax.set_title(f"{cell}    skew {skew:+.2f}", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Clean in-distribution score distributions, 9,000 CIFAR-10 test images\n"
        "The two full-covariance cells carry a right tail. The two diagonal cells do not.",
        fontsize=11)
    fig.supxlabel("score (squared Mahalanobis distance)", fontsize=9)
    fig.tight_layout()
    path = OUT / "score-distributions-four-cells.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_two(frame):
    """The naive cell against SVHN and against Texture, which is the cancellation made visible."""
    ident = frame[frame.split == "id_test"]["marginal_diagonal"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (dataset, auroc) in zip(axes, [("svhn", 12.90), ("texture", 94.29)]):
        other = frame[frame.split == dataset]["marginal_diagonal"].to_numpy(float)
        bins = np.linspace(min(ident.min(), other.min()), max(ident.max(), other.max()), 120)
        ax.hist(ident, bins=bins, density=True, alpha=0.6, label="in-distribution", color="#4C72B0")
        ax.hist(other, bins=bins, density=True, alpha=0.6, label=dataset, color="#C44E52")
        ax.set_title(f"marginal_diagonal on {dataset}    AUROC {auroc:.2f}", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(
        "The same scorer on two far-OOD datasets. On SVHN it separates backwards.\n"
        "Averaging these two is what produces a far-OOD figure near 61.", fontsize=11)
    fig.supxlabel("score (higher means more out-of-distribution)", fontsize=9)
    fig.tight_layout()
    path = OUT / "score-distributions-cancellation.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    frame = load(["split"] + CELLS)
    print("wrote", figure_one(frame).name)
    print("wrote", figure_two(frame).name)


if __name__ == "__main__":
    main()
