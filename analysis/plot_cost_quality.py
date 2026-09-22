"""The cost and quality trade-off as one curve, from cost-quality-pareto.csv.

A working figure, not a thesis figure. The thesis figure set has its own style module
and this deliberately does not import it, so that a later restyle is a rewrite of one
plotting call rather than a merge. What it is for is reading the frontier,
which a two-column table does not let anyone do.

The cost axis is logarithmic because the methods span two orders of magnitude, and
the AUROC axis is clipped at 80 because the diagonal-covariance cells fall so far
below the rest that a shared linear axis would compress everything else to a line.
Points below the clip are annotated with their value rather than dropped.

Run derive_cost_quality.py first. Writes one PNG beside this script.
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent
TABLE = OUT / "cost-quality-pareto.csv"
FLOOR = 80.0
PANELS = [("standard", "near_ood"), ("standard", "far_ood"),
          ("full_spectrum", "near_ood"), ("full_spectrum", "far_ood")]
#: Shortened only for the plot. Every table keeps the schema name.
SHORT = {
    "marginal_diagonal": "marg-diag",
    "marginal_full": "marg-full",
    "class_conditional_diagonal": "cc-diag",
    "class_conditional_full": "cc-full",
    "knn_normalized": "kNN-norm",
    "knn_unnormalized": "kNN-unnorm",
    "pca_residual_all_id": "res-allID",
    "pca_residual_all_id_l2": "res-allID-L2",
    "pca_residual_class_mean": "res-classmean",
    "pca_residual_class_mean_l2": "res-classmean-L2",
    "pca_residual_class_mean_whitened": "res-classmean-white",
}


def main():
    frame = pd.read_csv(TABLE)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (protocol, group) in zip(axes.ravel(), PANELS):
        block = frame[(frame.protocol == protocol) & (frame.group == group)].copy()
        block["plotted"] = block.auroc.clip(lower=FLOOR)
        front = block[block.on_pareto_frontier == "yes"].sort_values(
            "score_microseconds_per_sample")
        ax.plot(front.score_microseconds_per_sample, front.plotted,
                color="#C44E52", lw=1.6, zorder=1, label="Pareto frontier")
        # Labels collide badly in the cheap cluster, where six methods sit inside
        # one decade of cost and a point of AUROC. Fanning the offsets by rank is
        # cruder than a real label solver and enough to make the panel readable.
        block = block.sort_values(["score_microseconds_per_sample", "plotted"])
        offsets = [(6, 5), (6, -9), (-6, 7), (-6, -11), (6, 12), (-6, -18)]
        for position, (_, row) in enumerate(block.iterrows()):
            on = row.on_pareto_frontier == "yes"
            ax.scatter(row.score_microseconds_per_sample, row.plotted,
                       s=54 if on else 30,
                       color="#C44E52" if on else "#4C72B0",
                       zorder=3 if on else 2)
            label = SHORT.get(row.scorer, row.scorer)
            if row.auroc < FLOOR:
                label = f"{label} ({row.auroc:.1f})"
            dx, dy = offsets[position % len(offsets)]
            ax.annotate(label, (row.score_microseconds_per_sample, row.plotted),
                        textcoords="offset points", xytext=(dx, dy), fontsize=6.5,
                        ha="left" if dx > 0 else "right",
                        color="#C44E52" if on else "#555555", zorder=4)
        ax.set_xscale("log")
        ax.set_ylim(FLOOR - 1.5, 100.4)
        ax.axhline(FLOOR, color="0.8", lw=0.8, ls=":")
        ax.set_title(f"{protocol}   {group}   "
                     f"{len(front)} of {len(block)} on the frontier", fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylabel("dataset-averaged AUROC")
    for ax in axes[1]:
        ax.set_xlabel("scoring cost, microseconds per sample (log scale, n = 1)")
    fig.suptitle(
        "Scoring cost against detection quality, thirteen scorers, shrunk arm\n"
        "Cost is one unreplicated wall-clock measurement per scorer on node "
        "one workstation, unreplicated: no repeats, no interval, numpy exact "
        "search rather than faiss.\nAUROC values below "
        f"{FLOOR:.0f} are plotted at the floor and annotated with their real value.",
        fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT / "cost-quality-frontier.png"
    fig.savefig(path, dpi=150)
    print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
